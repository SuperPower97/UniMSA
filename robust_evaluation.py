import os
import torch
import yaml
import ipdb
import argparse
import numpy as np
from sklearn.manifold import TSNE
import torch.nn.functional as Func
from matplotlib import gridspec
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from core.dataset import MMDataEvaluationLoader
from models.UniMSA import build_model
from core.metric import MetricsTop


# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

parser = argparse.ArgumentParser() 
parser.add_argument('--config_file', type=str, default='') 
parser.add_argument('--key_eval', type=str, default='') 
opt = parser.parse_args()
print(opt)


def main():
    config_file = 'configs/eval_mosei.yaml' if opt.config_file == '' else opt.config_file
    
    with open(config_file) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    print(args)
    
    dataset_name = args['dataset']['datasetName']
    key_eval = args['base']['key_eval'] if opt.key_eval == '' else opt.key_eval
    lambda_ = args['base']['lambda']

    model = build_model(args).to(device)
    # ipdb.set_trace()
    metrics = MetricsTop(train_mode = args['base']['train_mode']).getMetics(dataset_name)
    missing_rate_list = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    Mult_acc_2 = []
    F1_score = []
    Non0_acc_2 = []
    Non0_F1 = []
    Mult_acc_3 = []
    Mult_acc_5 = []
    Mult_acc_7 = []
    MAE_a = []
    Corr = []
    for cur_r in missing_rate_list:
        test_results_list = []
        if dataset_name == 'sims':
            # for _, cur_seed  in enumerate([1111, 1112, 1113]):
            for _, cur_seed  in enumerate([1111]):
                best_ckpt = os.path.join(f'ckpt/{dataset_name}/lambda_{lambda_}/best_{key_eval}_{cur_seed}.pth')
                
                model.load_state_dict(torch.load(best_ckpt, weights_only=True)['state_dict'])
                args['base']['missing_rate_eval_test'] = cur_r # Set missing rate
                dataLoader = MMDataEvaluationLoader(args)
                test_results_cur_seed = evaluate(model, dataLoader, metrics, cur_r, dataset_name)
                test_results_list.append(test_results_cur_seed)

            if key_eval == 'Mult_acc_2':
                Mult_acc_2_avg = (test_results_list[0]['Mult_acc_2'] ) 
                F1_score_avg = (test_results_list[0]['F1_score'] ) 
                Mult_acc_2.append(Mult_acc_2_avg)
                F1_score.append(F1_score_avg)
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_2_avg: {Mult_acc_2_avg}, F1_score_avg: {F1_score_avg}')
            elif key_eval == 'Mult_acc_3':
                Mult_acc_3_avg = (test_results_list[0]['Mult_acc_3'] ) 
                Mult_acc_3.append(Mult_acc_3_avg)
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_3_avg: {Mult_acc_3_avg}')
            elif key_eval == 'Mult_acc_5':
                Mult_acc_5_avg = (test_results_list[0]['Mult_acc_5'] ) 
                Mult_acc_5.append(Mult_acc_5_avg)
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_5_avg: {Mult_acc_5_avg}')
            elif key_eval == 'MAE':
                MAE_avg = (test_results_list[0]['MAE'] ) 
                Corr_avg = (test_results_list[0]['Corr'] ) 
                MAE_a.append(MAE_avg)
                Corr.append(Corr_avg)
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, MAE_avg: {MAE_avg}, Corr_avg: {Corr_avg}')

        else:
            # for _, cur_seed  in enumerate([1111, 1112, 1113]):
            for _, cur_seed  in enumerate([1111]):
                best_ckpt = os.path.join(f'ckpt/{dataset_name}/lambda_{lambda_}/best_{key_eval}_{cur_seed}.pth')
                model.load_state_dict(torch.load(best_ckpt, weights_only=True)['state_dict'])
                args['base']['missing_rate_eval_test'] = cur_r # Set missing rate

                dataLoader = MMDataEvaluationLoader(args)
        
                test_results_cur_seed = evaluate(model, dataLoader, metrics, cur_r, dataset_name)
                
                test_results_list.append(test_results_cur_seed)

            if key_eval == 'Has0_acc_2':
                Has0_acc_2_avg = (test_results_list[0]['Has0_acc_2'] ) 
                Has0_F1_score_avg = (test_results_list[0]['Has0_F1_score'] ) 
                Mult_acc_2.append(Has0_acc_2_avg)
                F1_score.append(Has0_F1_score_avg)
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_2_avg: {Has0_acc_2_avg}, F1_score_avg: {Has0_F1_score_avg}')
            elif key_eval == 'Non0_acc_2':
                Non0_acc_2_avg = (test_results_list[0]['Non0_acc_2'] ) 
                Non0_F1_score_avg = (test_results_list[0]['Non0_F1_score'] ) 
                Non0_acc_2.append(Non0_acc_2_avg)
                Non0_F1.append(Non0_F1_score_avg)
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Non0_acc_2_avg: {Non0_acc_2_avg}, Non0_F1_score_avg: {Non0_F1_score_avg}')
            elif key_eval == 'Mult_acc_5':
                Mult_acc_5_avg = (test_results_list[0]['Mult_acc_5'] ) 
                Mult_acc_5.append(Mult_acc_5_avg)
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_5_avg: {Mult_acc_5_avg}')
            elif key_eval == 'Mult_acc_7':
                Mult_acc_7_avg = (test_results_list[0]['Mult_acc_7'] ) 
                Mult_acc_7.append(Mult_acc_7_avg)
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_7_avg: {Mult_acc_7_avg}')
            elif key_eval == 'MAE':
                MAE_avg = (test_results_list[0]['MAE'] ) 
                Corr_avg = (test_results_list[0]['Corr'] ) 
                MAE_a.append(MAE_avg)
                Corr.append(Corr_avg)
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, MAE_avg: {MAE_avg}, Corr_avg: {Corr_avg}')
            
    if key_eval == 'Has0_acc_2':
        Mult_acc_2_all_avg = sum(Mult_acc_2) / len(Mult_acc_2)
        Has0_F1_score_all_avg = sum(F1_score) / len(F1_score)
        print(f'key_eval: {key_eval}, Mult_acc_2: {Mult_acc_2_all_avg}, F1_score: {Has0_F1_score_all_avg}')
    elif key_eval == 'Non0_acc_2':
        Non0_acc_2_all_avg = sum(Non0_acc_2) / len(Non0_acc_2)
        Non0_F1_all_avg = sum(Non0_F1) / len(Non0_F1)
        print(f'key_eval: {key_eval}, Non0_acc_2: {Non0_acc_2_all_avg}, Non0_F1: {Non0_F1_all_avg}')
    elif key_eval == 'Mult_acc_2':
        Mult_acc_2_all_avg = sum(Mult_acc_2) / len(Mult_acc_2)
        F1_score_all_avg = sum(F1_score) / len(F1_score)
        print(f'key_eval: {key_eval}, Mult_acc_2: {Mult_acc_2_all_avg}, F1_score: {F1_score_all_avg}')
    elif key_eval == 'Mult_acc_3':
        Mult_acc_3_all_avg = sum(Mult_acc_3) / len(Mult_acc_3)
        print(f'key_eval: {key_eval}, Mult_acc_3: {Mult_acc_3_all_avg}')
    elif key_eval == 'Mult_acc_5':
        Mult_acc_5_all_avg = sum(Mult_acc_5) / len(Mult_acc_5)
        print(f'key_eval: {key_eval}, Mult_acc_5: {Mult_acc_5_all_avg}')
    elif key_eval == 'Mult_acc_7':
        Mult_acc_7_all_avg = sum(Mult_acc_7) / len(Mult_acc_7)
        print(f'key_eval: {key_eval}, Mult_acc_7: {Mult_acc_7_all_avg}')
    elif key_eval == 'MAE':
        MAE_a_all_avg = sum(MAE_a) / len(MAE_a)
        Corr_all_avg = sum(Corr) / len(Corr)
        print(f'key_eval: {key_eval}, MAE: {MAE_a_all_avg}, Corr_avg: {Corr_all_avg}')


def square_sum(gamma1,gamma):
    out = ((gamma1-gamma)**2).sum(dim=1, keepdim=True) 
    return out 

def fuse_nig(gamma1, v1, alpha1, beta1, gamma2, v2, alpha2, beta2):
    # Eq. 16
    gamma = (gamma1*v1 + gamma2*v2) / (v1+v2 + 1e-12)
    v = v1 + v2
    alpha = alpha1 + alpha2 + 0.5
    beta = beta1 + beta2 + 0.5 * (v1 * square_sum(gamma1, gamma) + v2 * square_sum(gamma2, gamma))
    return gamma, v, alpha, beta

def uncertainty_score(alpha, beta, v):

    uncertainty = (beta / v * (alpha - 1))

    return uncertainty

def evaluate(model, eval_loader, metrics, cur_r, dataset_nam):

    y_pred, y_true = [], []
    y_pos, y_neg = [], []
    y_pred_g, y_v_g, y_alpha_g, y_beta_g, y_pred_a, y_v_a, y_alpha_a, y_beta_a = [], [], [], [], [], [], [], []
    model.eval()
    for cur_iter, data in enumerate(eval_loader):
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))
        sentiment_labels = data['labels']['M'].to(device)

        with torch.no_grad():
            out = model((None, None, None), incomplete_input)
  
        y_pred_g.append(out['sentiment_preds']['gamma_g'].cpu())
        y_v_g.append(out['sentiment_preds']['v_g'].cpu())
        y_alpha_g.append(out['sentiment_preds']['v_g'].cpu())
        y_beta_g.append(out['sentiment_preds']['beta_g'].cpu())
        
        y_pred_a.append(out['sentiment_preds']['gamma_a'].cpu())
        y_v_a.append(out['sentiment_preds']['v_a'].cpu())
        y_alpha_a.append(out['sentiment_preds']['v_a'].cpu())
        y_beta_a.append(out['sentiment_preds']['beta_a'].cpu())

        y_pos.append(out['sentiment_preds']['pos'].cpu())
        y_neg.append(out['sentiment_preds']['neg'].cpu())
        y_true.append(sentiment_labels.cpu())
    pred_g, v_g, alpha_g, beta_g = torch.cat(y_pred_g), torch.cat(y_v_g), torch.cat(y_alpha_g), torch.cat(y_beta_g)
    pred_a, v_a, alpha_a, beta_a = torch.cat(y_pred_a), torch.cat(y_v_a), torch.cat(y_alpha_a), torch.cat(y_beta_a)
    pred_f, v_f, alpha_f, beta_f = fuse_nig(pred_g, v_g, alpha_g, beta_g, pred_a, v_a, alpha_a, beta_a)  

    true = torch.cat(y_true)

    results = metrics(pred_f, true)

    return results



if __name__ == '__main__':
    main()
