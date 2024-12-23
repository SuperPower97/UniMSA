# Decoding Emotional Silences: Reliable Multimodal Sentiment Analysis with Bipolar Uncertainty


## Content
- [Data Preparation](#Data-preparation)
- [Environment](#Environment)
- [Training](#Training)
- [Evaluation](#Evaluation)


## Data Preparation
MOSI/MOSEI/CH-SIMS Download: Please see [MMSA](https://github.com/thuiar/MMSA)

## Environment
The basic training environment for the results in the paper is Python 3.12.7, Pytorch 2.5.1, with NVIDIA 4090D. 

## Training
You can quickly run the code with the following command:
```
python run_mosei_different_seeds.py
python run_mosi_different_seeds.py
python run_sims_different_seeds.py
```

## Evaluation
```
CUDA_VISIBLE_DEVICES=0 python robust_evaluation.py --config_file configs/eval_mosi.yaml --key_eval Has0_acc_2
```

