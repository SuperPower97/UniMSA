import ipdb
from torch import nn
from torch.nn import functional as F
from .utils import fuse_nig, square_sum, NIG_loss

class MultimodalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.alpha = args['base']['alpha']
        self.beta = args['base']['beta']
        self.gamma = args['base']['gamma']
        self.sigma = args['base']['sigma']
        self.lambda1 = args['base']['lambda']
        self.CE_Fn = nn.CrossEntropyLoss()
        self.MSE_Fn = nn.MSELoss() 


    def forward(self, out, label):

        l_cc = self.MSE_Fn(out['w'], label['completeness_labels']) if out['w'] is not None else 0

        l_adv = self.CE_Fn(out['effectiveness_discriminator_out'], label['effectiveness_labels']) if out['effectiveness_discriminator_out'] is not None else 0

        l_rec = self.MSE_Fn(out['rec_feats'], out['complete_feats']) if out['rec_feats'] is not None and out['complete_feats'] is not None else 0

        y_pred_g, v_g, alpha_g, beta_g = out['sentiment_preds']['gamma_g'], out['sentiment_preds']['v_g'], out['sentiment_preds']['alpha_g'], out['sentiment_preds']['beta_g']
        y_pred_a, v_a, alpha_a, beta_a = out['sentiment_preds']['gamma_a'], out['sentiment_preds']['v_a'], out['sentiment_preds']['alpha_a'], out['sentiment_preds']['beta_a']
        
        # fuse multi views
        y_pred_f, v_f, alpha_f, beta_f = fuse_nig(y_pred_g, v_g, alpha_g, beta_g, y_pred_a, v_a, alpha_a, beta_a)

        l_sp = self.MSE_Fn(y_pred_f, label['sentiment_labels'])

        l_nig = NIG_loss(y_pred_f, v_f, alpha_f, beta_f, l_sp, coeffi=self.lambda1) + \
                NIG_loss(y_pred_a, v_a, alpha_a, beta_a, l_sp, coeffi=self.lambda1) +\
                NIG_loss(y_pred_g, v_g, alpha_g, beta_g, l_sp, coeffi=self.lambda1)
                
        loss = self.alpha * l_cc + self.beta * l_adv + self.gamma * l_rec + self.sigma * l_nig

        return {'loss': loss, 'l_nig': l_nig, 'l_cc': l_cc, 'l_adv': l_adv, 'l_rec': l_rec, 'l_sp': l_sp}, y_pred_f
    

    def wo_NIG(self, out, label):

        l_cc = self.MSE_Fn(out['w'], label['completeness_labels']) if out['w'] is not None else 0

        l_adv = self.CE_Fn(out['effectiveness_discriminator_out'], label['effectiveness_labels']) if out['effectiveness_discriminator_out'] is not None else 0

        l_rec = self.MSE_Fn(out['rec_feats'], out['complete_feats']) if out['rec_feats'] is not None and out['complete_feats'] is not None else 0

        y_pred = out['sentiment_preds']['preds']

        l_sp = self.MSE_Fn(y_pred, label['sentiment_labels'])
        
        loss = self.alpha * l_cc + self.beta * l_adv + self.gamma * l_rec + self.sigma * l_sp

        return {'loss': loss, 'l_sp': l_sp, 'l_cc': l_cc, 'l_adv': l_adv, 'l_rec': l_rec}

