import torch
import ipdb
from torch import nn, einsum
from einops import rearrange, repeat

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

class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm_qkv(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        return self.fn(q, k, v)

class PreNorm_hyper(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, h_dominate, h_a, h_v, h_hyper):
        h_dominate = self.norm1(h_dominate)
        h_a = self.norm2(h_a)
        h_v = self.norm3(h_v)
        h_hyper = self.norm4(h_hyper)

        return self.fn(h_dominate, h_a, h_v, h_hyper)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class HhyperLearningLayer(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_ta = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_tv = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_ta = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_tv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=True),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, h_dominate, h_a, h_v, h_hyper):
        h = self.heads

        q = self.to_q(h_dominate)
        k_a = self.to_k_ta(h_a)
        k_v = self.to_k_tv(h_v)

        v_a = self.to_v_ta(h_a)
        v_v = self.to_v_tv(h_v)

        q, k_a, k_v, v_a, v_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k_a, k_v, v_a, v_v))

        dots_q_ka = einsum('b h i d, b h j d -> b h i j', q, k_a) * self.scale
        attn_q_ka = self.attend(dots_q_ka)
        out_q_ka = einsum('b h i j, b h j d -> b h i d', attn_q_ka, v_a)
        out_q_ka = rearrange(out_q_ka, 'b h n d -> b n (h d)')

        dots_q_kv = einsum('b h i d, b h j d -> b h i j', q, k_v) * self.scale
        attn_q_kv = self.attend(dots_q_kv)
        out_q_kv = einsum('b h i j, b h j d -> b h i d', attn_q_kv, v_v)
        out_q_kv = rearrange(out_q_kv, 'b h n d -> b n (h d)')

        h_hyper_shift = self.to_out(out_q_ka + out_q_kv)
        h_hyper += h_hyper_shift

        return h_hyper


class HhyperLearningEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_hyper(dim, HhyperLearningLayer(dim, heads = heads, dim_head = dim_head, dropout = dropout))
            ]))

    def forward(self, h_domonate_list, h_a, h_v, h_hyper):
        for i, attn in enumerate(self.layers):
            h_hyper = attn[0](h_domonate_list[i], h_a, h_v, h_hyper)
        return h_hyper


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    
    def forward(self, x, save_hidden=False):
        if save_hidden == True:
            hidden_list = []
            hidden_list.append(x)
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
                hidden_list.append(x)
            return hidden_list
        else:
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
            return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm_qkv(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, tgt, memory):
        for attn1, attn2, ff in self.layers:
            tgt = attn1(tgt, tgt, tgt) + tgt
            tgt = attn1(tgt, memory, memory) + tgt
            tgt = ff(tgt) + tgt
        return tgt



class CrossTransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, source_x, target_x):
        for attn, ff in self.layers:
            target_x_tmp = attn(target_x, source_x, source_x)
            target_x = target_x_tmp + target_x
            target_x = ff(target_x) + target_x
        return target_x



class Transformer(nn.Module):
    def __init__(self, *, num_frames, token_len, save_hidden, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.token_len = token_len
        self.save_hidden = save_hidden

        if token_len is not None:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + token_len, dim))
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
        else:
             self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
             self.extra_token = None

        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()


    def forward(self, x):
        b, n, _ = x.shape

        if self.token_len is not None:
            extra_token = repeat(self.extra_token, '1 n d -> b n d', b = b)
            x = torch.cat((extra_token, x), dim=1)
            x = x + self.pos_embedding[:, :n+self.token_len]
        else:
            x = x + self.pos_embedding[:, :n]

        x = self.dropout(x)
        x = self.encoder(x, self.save_hidden)

        return x


class CrossTransformer(nn.Module):
    def __init__(self, *, source_num_frames, tgt_num_frames, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.pos_embedding_s = nn.Parameter(torch.randn(1, source_num_frames + 1, dim))
        self.pos_embedding_t = nn.Parameter(torch.randn(1, tgt_num_frames + 1, dim))
        self.extra_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.CrossTransformerEncoder = CrossTransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

    def forward(self, source_x, target_x):
        b, n_s, _ = source_x.shape
        b, n_t, _ = target_x.shape

        extra_token = repeat(self.extra_token, '1 1 d -> b 1 d', b = b)

        source_x = torch.cat((extra_token, source_x), dim=1)
        source_x = source_x + self.pos_embedding_s[:, : n_s+1]

        target_x = torch.cat((extra_token, target_x), dim=1)
        target_x = target_x + self.pos_embedding_t[:, : n_t+1]

        source_x = self.dropout(source_x)
        target_x = self.dropout(target_x)
        x_s2t = self.CrossTransformerEncoder(source_x, target_x)

        return x_s2t


class CrossBipolarAttention(nn.Module):
    def __init__(self, heads, hidden_dim):
        super(CrossBipolarAttention, self).__init__()
        self.hidden_dim = hidden_dim

        # 嵌入层
        self.visual_embedding = nn.Linear(hidden_dim, hidden_dim)
        self.textual_embedding = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()

        # 注意力层
        self.dual_attention_linear_1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dual_attention_linear_2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.cross_modal_linear_1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.cross_modal_linear_2 = nn.Linear(hidden_dim * 2, hidden_dim)

        # 预测层
        self.predict_linear = nn.Linear(hidden_dim * 2, 1)
        self.add_attention_linear_1 = nn.Linear(hidden_dim, heads)
        self.add_attention_linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.add_attention_linear_3 = nn.Linear(hidden_dim, hidden_dim)
        
        # 可训练的参数
        self.add_attention_matrix_1 = nn.Parameter(torch.randn(1, hidden_dim))
        self.add_attention_matrix_2 = nn.Parameter(torch.randn(1, hidden_dim))
    

    def forward(self, source_x, target_x):
        # 假设 source_x 和 target_x 的形状都是 (B, 8, hidden_dim)
        # source_x_emb = self.visual_embedding(source_x)
        # source_x_emb = self.tanh(source_x_emb)
        # target_x_emb = self.textual_embedding(target_x)
        # target_x_emb = self.tanh(target_x_emb)
        # ipdb.set_trace()
        # 计算注意力
        S = self.tanh(self.add_attention_linear_1(source_x + target_x))         # (B, 8, 8)
        T_p = torch.matmul(torch.softmax(S, dim=1), target_x)                   # (B, 8, 8) * (B, 8, hidden_dim) = (B, 8, hidden_dim)
        V_p = torch.matmul(torch.softmax(S.transpose(1, 2), dim=1), source_x)   # (B, 8, 8) * (B, 8, hidden_dim) = (B, 8, hidden_dim)
        T_n = torch.matmul(-0.6 * torch.softmax(S, dim=1), target_x)
        V_n = torch.matmul(-0.6 * torch.softmax(S.transpose(1, 2), dim=1), source_x)
        Pos = torch.softmax(S, dim=1)
        Neg = -0.6 * torch.softmax(S, dim=1)
        # 跨模态融合
        T_star = self.dual_attention_linear_1(torch.cat([T_p, T_n], dim=2))     # [B, 8, hidden_dim]
        V_star = self.dual_attention_linear_2(torch.cat([V_p, V_n], dim=2))     # [B, 8, hidden_dim]
        T_star = self.tanh(T_star)
        V_star = self.tanh(V_star)

        V_f = self.cross_modal_linear_1(torch.cat([source_x, T_star], dim=2))   # [B, 8, hidden_dim]
        T_f = self.cross_modal_linear_2(torch.cat([target_x, V_star], dim=2))   # [B, 8, hidden_dim]
        V_f = self.tanh(V_f)
        T_f = self.tanh(T_f)

        # 最终输出
        # alpha_v = self.tanh(self.add_attention_linear_2(V_f + self.add_attention_matrix_1)) # [B, 8, hidden_dim]
        # V_f_star = torch.matmul(alpha_v.transpose(1, 2), V_f)                               # [B, 8, ]
        # alpha_t = self.tanh(self.add_attention_linear_3(T_f + self.add_attention_matrix_2)) # [B, 8, hidden_dim]
        # T_f_star = torch.matmul(alpha_t.transpose(1, 2), T_f)
        # output = self.predict_linear(torch.cat([V_f_star, T_f_star], dim=2))
        # output = output.squeeze(2)

        alpha_v = self.tanh(self.add_attention_linear_2(V_f + self.add_attention_matrix_1)) # [B, 8, hidden_dim]
        alpha_t = self.tanh(self.add_attention_linear_3(T_f + self.add_attention_matrix_2)) # [B, 8, hidden_dim]
        output = alpha_v + alpha_t  # [B, 8, hidden_dim]
        # ipdb.set_trace()
        return output, Pos, Neg
