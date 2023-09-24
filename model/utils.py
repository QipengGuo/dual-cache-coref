from coref_utils.utils import get_mention_to_cluster_idx
import torch
from logger import *
def get_gt_actions(pred_mentions, document, mem_type_config, ment_scores=None, train=False, max_ent=-1, cache_mode='lru'):
    if "clusters" in document:
        # Ground truth is avaliable
        gt_clusters = document["clusters"]
        if train:
            return get_actions_ours(pred_mentions, gt_clusters, max_ent, ment_scores, cache_mode=cache_mode)
        else:
            return get_actions_unbounded_fast(pred_mentions, gt_clusters)
    else:
        # Don't have ground truth clusters i.e. running it in the wild
        # Generate dummy actions
        return [(-1, "i")] * len(pred_mentions)


def action_sequences_to_clusters(actions, mentions, doc=None, ask_cor=False):
    clusters = []
    cell_to_clusters = {}
    actions1 = []
    if len(actions)>len(mentions):
        actions1 = actions[len(mentions):]

    ment_to_clusters = {}
    ment_idx = 0
    show_idx = []
    for mention, (cell_idx, action_type) in zip(mentions, actions):
        if action_type == "c":
            cell_to_clusters[cell_idx].append(mention)
            ment_to_clusters[ment_idx] = cell_idx
        elif action_type == "o":
            # Overwrite
            if cell_idx in cell_to_clusters:
                # Remove the old cluster and initialize the new
                clusters.append(cell_to_clusters[cell_idx])
            cell_to_clusters[cell_idx] = [mention]
            ment_to_clusters[ment_idx] = cell_idx
        elif action_type == "n":
            clusters.append([mention])
        elif action_type == 'v':
            tt = ' '.join(['Old V', ts(mention), '|', ts(cell_to_clusters[cell_idx])])
            show_idx.append([ment_idx, tt])
        ment_idx += 1
    correct_cnt = 0 
    if len(actions1)>0:
        ment_idx = 0
        for mention, (cell_idx, action_type) in zip(mentions, actions1):
            if action_type == "c":
                if ment_idx in ment_to_clusters:
                    cell_to_clusters[cell_idx].extend(cell_to_clusters[ment_to_clusters[ment_idx]])
                else:
                    cell_to_clusters[cell_idx].append(mention)
                correct_cnt += 1
                #print('st add', cell_idx)
            elif action_type == "o":
                # Overwrite
                if cell_idx in cell_to_clusters:
                    # Remove the old cluster and initialize the new
                    clusters.append(cell_to_clusters[cell_idx])
                if ment_idx in ment_to_clusters:
                    cell_to_clusters[cell_idx].extend(cell_to_clusters[ment_to_clusters[ment_idx]])
                else:
                    cell_to_clusters[cell_idx] = [mention]
            elif action_type == "n":
                clusters.append([mention])
            ment_idx += 1

    for cell_idx, cluster in cell_to_clusters.items():
        clusters.append(cluster)
    if ask_cor:
        print(correct_cnt)
        return clusters, correct_cnt
    else:
        return clusters

def get_actions_unbounded_fast(pred_mentions, gt_clusters):
    actions = []
    cell_counter = 0
    cluster_to_cell = {}
    mention_to_cluster = get_mention_to_cluster_idx(gt_clusters)

    for idx, mention in enumerate(pred_mentions):
        if tuple(mention) not in mention_to_cluster:
            actions.append((-1, "i"))
        else:
            mention_cluster = mention_to_cluster[tuple(mention)]
            if mention_cluster in cluster_to_cell:
                # Cluster is already being tracked
                actions.append((cluster_to_cell[mention_cluster], "c"))
            else:
                # Cluster is not being tracked
                # Add the mention to being tracked
                cluster_to_cell[mention_cluster] = cell_counter
                actions.append((cell_counter, "o"))
                cell_counter += 1

    return actions


def get_actions_ours(pred_mentions, gt_clusters, max_ents, ment_scores=None, cache_mode='lru'):
    def rindex(x, y):
        ret = None
        for i, _x in enumerate(x):
            if _x==y:
                ret = i
        return ret
    pred_mentions = [tuple(mention) for mention in pred_mentions]

    # Useful data structures
    mention_to_cluster = get_mention_to_cluster_idx(gt_clusters)

    actions = []
    cell_to_cluster = {}
    cell_to_last_used = [0 for cell in range(max_ents)]  # Initialize last usage of cell
    cluster_to_cell = {}

    # Initialize with all the mentions
    cluster_to_rem_mentions = [len(cluster) for cluster in gt_clusters]
    lru_list = list(range(max_ents))
    pred_actions = []  # argmax actions

    # Boolean to track if we have started tracking any entities
    # This value can be false if we are processing subsequent chunks of a long document
    first_overwrite: bool = True
    MAX_ENT = max_ents
    prev_idx = []
    mems = []
    ENT_LRU = int(cache_mode.split('_')[0])
    ENT_LFU = int(cache_mode.split('_')[1])
    if len(ment_scores)==1:
        ment_scores=ment_scores[0]
    last_access = []
    last_access.append(0)
    for ment_idx, ment in enumerate(pred_mentions):
        z_flag = True if ment_scores[ment_idx]<0.0 else False
        if first_overwrite:
            pred_cell_idx, pred_action_str = 0, "o"
            prev_idx.append([])
        else:
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
            kmem = [mems[x] for x in idx]
            if mention_to_cluster.get(ment, 'Another None') in kmem:
                pred_cell_idx, pred_action_str = rindex(kmem, mention_to_cluster[ment]), 'c'
            else:
                pred_cell_idx, pred_action_str = idx.shape[0], 'o'
        pred_actions.append((pred_cell_idx, pred_action_str))
        if first_overwrite:
            first_overwrite = False
            # We start with a single empty memory cell
            ent_counter = torch.tensor([0.0 if z_flag else 1.0])
            mems.append(mention_to_cluster.get(ment, None))
        else:
            if pred_action_str == "c":
                # Perform coreference update on the cluster referenced by pred_cell_idx
                ent_counter[idx[pred_cell_idx]] = ent_counter[idx[pred_cell_idx]] + (0 if z_flag else 1)
                last_access.append(idx[pred_cell_idx].item())
            elif pred_action_str == "o":
                # Append the new entity to the entity cluster array
                ent_counter = torch.cat(
                    [ent_counter, torch.tensor([0.0 if z_flag else 1.0])], dim=0
                )
                mems.append(mention_to_cluster.get(ment, None))
                last_access.append(len(mems)-1)
    return pred_actions[1:]
