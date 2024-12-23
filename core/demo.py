import torch
import ipdb
import torch.nn as nn


checkpoint_path = './ckpt/mosi/lambda_0.1/best_Corr_1111.pth'  # 替换为您的 .pth 文件路径
check = torch.load(checkpoint_path)
keys = check['state_dict'].keys()
ipdb.set_trace()