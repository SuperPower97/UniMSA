import yaml
import subprocess


yaml_file = '/home/icdm/wyt/EUCSA-fus/configs/train_mosei.yaml'  

seeds = [1111, 1112, 1113]
lambda_values = [0.1]
for seed in seeds:
    for new_lambda in lambda_values:
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        config['base']['seed'] = seed
        config['base']['lambda'] = new_lambda
        with open(yaml_file, 'w') as file:
            yaml.dump(config, file)
        print(f"current seed: {seed}")
        print(f"current lambda: {new_lambda}")
        command = f"CUDA_VISIBLE_DEVICES=0 python train.py --config_file {yaml_file}"
        subprocess.run(command, shell=True)