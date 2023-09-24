import torch
from model.memory import BaseMemory
from pytorch_utils.modules import MLP
import torch.nn as nn

from omegaconf import DictConfig
from typing import Dict, Tuple, List
from torch import Tensor


class EntityMemory(BaseMemory):
    """Module for clustering proposed mention spans using Entity-Ranking paradigm."""

    def __init__(
        self, config: DictConfig, span_emb_size: int, drop_module: nn.Module
    ) -> None:
        super(EntityMemory, self).__init__(config, span_emb_size, drop_module)
        self.mem_type: DictConfig = config.mem_type

    def forward(
        self,
        ment_boundaries: Tensor,
        mention_emb_list: Tensor,
        metadata: Dict,
        memory_init: Dict = None,
        ment_scores = None,
        cache_mode=None,
    ) -> Tuple[List[Tuple[int, str]], Dict]:
        """Forward pass for clustering entity mentions during inference/evaluation.

        Args:
         ment_boundaries: Start and end token indices for the proposed mentions.
         mention_emb_list: Embedding list of proposed mentions
         metadata: Metadata features such as document genre embedding
         memory_init: Initializer for memory. For streaming coreference, we can pass the previous
                  memory state via this dictionary

        Returns:
                pred_actions: List of predicted clustering actions.
                mem_state: Current memory state.
        """

        # Initialize memory
        if memory_init is not None:
            mem_vectors, ent_counter, last_mention_start = self.initialize_memory(
                **memory_init
            )
            mem_ment_scores = memory_init['mem_ment_scores']
        else:
            mem_vectors, ent_counter, last_mention_start = self.initialize_memory()

        pred_actions = []  # argmax actions

        # Boolean to track if we have started tracking any entities
        # This value can be false if we are processing subsequent chunks of a long document
        first_overwrite: bool = True if torch.sum(ent_counter) == 0 else False
        ENT_LRU = int(cache_mode.split('_')[0])
        ENT_LFU = int(cache_mode.split('_')[1])
        prev_idx = []
        prev_feat = []
        last_access = []
        mem_start = len(ent_counter)-len(mem_vectors)
        mem_last = 0
        if len(ent_counter)>1:
            idx = last_mention_start.argsort()
            last_access.extend(idx.cpu().numpy().tolist())
        else:
            last_access.append(0)
        for ment_idx, ment_emb in enumerate(mention_emb_list):
            ment_start, ment_end = ment_boundaries[ment_idx]
            z_flag = True if ment_scores[ment_idx]<0.0 else False
            feature_embs = self.get_feature_embs(
                ment_start, last_mention_start, ent_counter, metadata
            )
            prev_feat.append(feature_embs)

            if first_overwrite:
                # First mention is always an overwrite
                pred_cell_idx, pred_action_str = 0, "o"
                prev_idx.append([])
            else:
                # Predict whether the mention is coreferent with existing clusters or a new cluster
                freq = {}
                for xx in last_access:
                    freq[xx] = freq.get(xx, 0)+1
                freq_list = [x[0] for x in list(sorted(freq.items(), key=lambda x:x[1]))]
                if ENT_LFU==0:
                    _last_access = []
                else:
                    _last_access = freq_list[-ENT_LFU:]
                _cnt = 0
                while len(_last_access) < ENT_LRU + ENT_LFU and len(last_access)-_cnt-1>=0:
                    t = last_access[len(last_access)-_cnt-1]
                    if t not in _last_access:
                        _last_access.append(t)
                    _cnt += 1
                mem_last = min(_last_access)
                idx = torch.LongTensor(_last_access).cuda()
                _mem_vectors, _ent_counter, _feature_embs = mem_vectors[idx-mem_start] , ent_counter[idx], feature_embs[idx]
                coref_new_scores = self.get_coref_new_scores(
                    ment_emb, _mem_vectors, _ent_counter, _feature_embs
                )
                pred_cell_idx, pred_action_str = self.assign_cluster(coref_new_scores)
                if z_flag: # c means reference a former item, z means adding to memory but it is not a singleton, o means adding to memory and it is a singleton
                    pred_cell_idx, pred_action_str = len(mem_vectors)+mem_start, 'z'
                if pred_action_str=='c':
                    pred_cell_idx = idx[pred_cell_idx].item()
                else:
                    pred_cell_idx = mem_vectors.shape[0]+mem_start

            pred_actions.append((pred_cell_idx, pred_action_str))

            if first_overwrite:
                first_overwrite = False
                # We start with a single empty memory cell
                mem_vectors = torch.unsqueeze(ment_emb, dim=0)
                ent_counter = torch.tensor([0.0 if z_flag else 1.0], device=self.device)
                mem_ment_scores = ment_scores[ment_idx].unsqueeze(0)
                last_mention_start[0] = ment_start
            else:
                if pred_action_str == "c":
                    # Perform coreference update on the cluster referenced by pred_cell_idx
                    coref_vec = self.coref_update(
                        ment_emb, mem_vectors, pred_cell_idx, ent_counter
                    )
                    mem_vectors[pred_cell_idx-mem_start] = coref_vec
                    ent_counter[pred_cell_idx] = ent_counter[pred_cell_idx] + (0 if z_flag else 1)
                    last_mention_start[pred_cell_idx] = ment_start
                    last_access.append(pred_cell_idx)

                elif pred_action_str == "o":
                    # Append the new entity to the entity cluster array
                    mem_vectors = torch.cat(
                        [mem_vectors, torch.unsqueeze(ment_emb, dim=0)], dim=0
                    )
                    ent_counter = torch.cat(
                        [ent_counter, torch.tensor([0.0 if z_flag else 1.0], device=self.device)], dim=0
                    )
                    last_mention_start = torch.cat(
                        [last_mention_start, ment_start.unsqueeze(dim=0)], dim=0
                    )
                    mem_ment_scores = torch.cat(
                        [mem_ment_scores, ment_scores[ment_idx].unsqueeze(dim=0)], dim=0
                    )
 
                    last_access.append(len(mem_vectors)-1+mem_start)
                elif pred_action_str == "z":
                    # Append the new entity to the entity cluster array
                    mem_vectors = torch.cat(
                        [mem_vectors, torch.unsqueeze(ment_emb, dim=0)], dim=0
                    )
                    ent_counter = torch.cat(
                        [ent_counter, torch.tensor([0.0], device=self.device)], dim=0
                    )
                    last_mention_start = torch.cat(
                        [last_mention_start, ment_start.unsqueeze(dim=0)], dim=0
                    )
                    mem_ment_scores = torch.cat(
                        [mem_ment_scores, ment_scores[ment_idx].unsqueeze(dim=0)], dim=0
                    )
 
                    last_access.append(len(mem_vectors)-1+mem_start)

        mem_state = {
            "mem": mem_vectors,
            "ent_counter": ent_counter,
            "last_mention_start": last_mention_start,
            "mem_ment_scores": mem_ment_scores,
        }
        return pred_actions, mem_state


    def forward_training(
        self,
        ment_boundaries: Tensor,
        mention_emb_list: Tensor,
        gt_actions, 
        metadata: Dict,
        memory_init: Dict = None,
        ment_scores = None,
        cache_mode=None,
    ) -> Tuple[List[Tuple[int, str]], Dict]:
        """Forward pass for clustering entity mentions during inference/evaluation.

        Args:
         ment_boundaries: Start and end token indices for the proposed mentions.
         mention_emb_list: Embedding list of proposed mentions
         metadata: Metadata features such as document genre embedding
         memory_init: Initializer for memory. For streaming coreference, we can pass the previous
                  memory state via this dictionary

        Returns:
                pred_actions: List of predicted clustering actions.
                mem_state: Current memory state.
        """

        # Initialize memory
        ret = []
        ENT_LRU = int(cache_mode.split('_')[0])
        ENT_LFU = int(cache_mode.split('_')[1])

        if memory_init is not None:
            mem_vectors, ent_counter, last_mention_start = self.initialize_memory(
                **memory_init
            )
            mem_ment_scores = memory_init['mem_ment_scores']
        else:
            mem_vectors, ent_counter, last_mention_start = self.initialize_memory()

        pred_actions = []  # argmax actions

        # Boolean to track if we have started tracking any entities
        # This value can be false if we are processing subsequent chunks of a long document
        first_overwrite: bool = True if torch.sum(ent_counter) == 0 else False
        ment_scores=ment_scores[0]
        prev_idx = []
        prev_feat = []
        last_access = [0]
        for ment_idx, ment_emb in enumerate(mention_emb_list):
            z_flag = True if ment_scores[ment_idx]<0.0 else False
            ment_start, ment_end = ment_boundaries[ment_idx]
            feature_embs = self.get_feature_embs(
                ment_start, last_mention_start, ent_counter, metadata
            )

            if first_overwrite:
                # First mention is always an overwrite
                pred_cell_idx, pred_action_str = 0, "o"
                prev_idx.append([])
            else:
                # Predict whether the mention is coreferent with existing clusters or a new cluster
                freq = {}
                for xx in last_access:
                    freq[xx] = freq.get(xx, 0)+1
                freq_list = [x[0] for x in list(sorted(freq.items(), key=lambda x:x[1]))]
                if ENT_LFU==0:
                    _last_access = []
                else:
                    _last_access = freq_list[-ENT_LFU:]
                _cnt = 0
                while len(_last_access) < ENT_LRU + ENT_LFU and len(last_access)-_cnt-1>=0:
                    t = last_access[len(last_access)-_cnt-1]
                    if t not in _last_access:
                        _last_access.append(t)
                    _cnt += 1

                idx = torch.LongTensor(_last_access).cuda()

                _mem_vectors, _ent_counter, _feature_embs = mem_vectors[idx] , ent_counter[idx], feature_embs[idx]
                coref_new_scores = self.get_coref_new_scores(
                    ment_emb, _mem_vectors, _ent_counter, _feature_embs
                )
                ret.append(coref_new_scores)
                pred_cell_idx, pred_action_str = gt_actions[ment_idx-1]
                if pred_action_str=='c':
                    pred_cell_idx = idx[pred_cell_idx].item()
                else:
                    pred_cell_idx = mem_vectors.shape[0]

            pred_actions.append((pred_cell_idx, pred_action_str))
            if first_overwrite:
                first_overwrite = False
                # We start with a single empty memory cell
                mem_vectors = torch.unsqueeze(ment_emb, dim=0)
                ent_counter = torch.tensor([0.0 if z_flag else 1.0], device=self.device)
                mem_ment_scores = ment_scores[ment_idx].unsqueeze(0)
                last_mention_start[0] = ment_start
            else:
                if pred_action_str == "c":
                    # Perform coreference update on the cluster referenced by pred_cell_idx
                    coref_vec = self.coref_update(
                        ment_emb, mem_vectors, pred_cell_idx, ent_counter
                    )
                    mask = torch.arange(start=0, end=len(mem_vectors), device=self.device)==pred_cell_idx
                    mask = mask[:,None].float()
                    mem_vectors = mem_vectors*(1.-mask) + mask * coref_vec
                    ent_counter[pred_cell_idx] = ent_counter[pred_cell_idx] + (0 if z_flag else 1)
                    last_mention_start[pred_cell_idx] = ment_start
                    #last_access.remove(pred_cell_idx)
                    last_access.append(pred_cell_idx)

                elif pred_action_str == "o":
                    # Append the new entity to the entity cluster array
                    mem_vectors = torch.cat(
                        [mem_vectors, torch.unsqueeze(ment_emb, dim=0)], dim=0
                    )
                    ent_counter = torch.cat(
                        [ent_counter, torch.tensor([0.0 if z_flag else 1.0], device=self.device)], dim=0
                    )
                    last_mention_start = torch.cat(
                        [last_mention_start, ment_start.unsqueeze(dim=0)], dim=0
                    )
                    mem_ment_scores = torch.cat(
                        [mem_ment_scores, ment_scores[ment_idx].unsqueeze(dim=0)], dim=0
                    )
 
                    last_access.append(len(mem_vectors)-1)
  
        return ret
