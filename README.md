# dual-cache-coref

Please follow the instructions in https://github.com/shtoshni/fast-coref to get data, model and configurations. 

Next, run 
```
python main.py experiment=litbank paths.model_dir=../models/joint_best/  model/doc_encoder/transformer=longformer_joint override_encoder=True train=False 
```

An example training command 
```
python -u main.py experiment=litbank model/doc_encoder/transformer=longformer_joint override_encoder=True train=True
```
