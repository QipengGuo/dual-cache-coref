env CUDA_VISIBLE_DEVICES=0 python -u main.py experiment=wikicoref paths.model_dir=joint_best/ model/doc_encoder/transformer=longformer_joint override_encoder=True train=False cache_mode="250_250"
