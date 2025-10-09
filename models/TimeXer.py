import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, num_classes, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window * num_classes)
        self.dropout = nn.Dropout(head_dropout)
        self.target_window = target_window
        self.num_classes = num_classes

    def forward(self, x):
        bs = x.size(0)
        x = self.flatten(x)               # [bs, nf]
        x = self.linear(x)                # [bs, L*C]
        x = self.dropout(x)
        return x.view(bs, self.target_window, self.num_classes)  # [bs, L, C]


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.size(1)
        glb = self.glb_token.repeat(x.size(0), 1, 1, 1)
        x = x.unfold(-1, self.patch_len, self.patch_len)
        x = x.reshape(x.size(0) * x.size(1), x.size(2), x.size(3))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = x.view(-1, n_vars, x.size(-2), x.size(-1))
        x = torch.cat([x, glb], dim=2)
        x = x.view(x.size(0) * x.size(1), x.size(2), x.size(3))
        return self.dropout(x), n_vars


class EncoderLayer(nn.Module):
    def __init__(self, self_attn, cross_attn, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attn(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)

        x_glb = x[:, -1:, :]
        B, _, D = x_glb.size()
        x_glb2 = x_glb.view(B, -1, D)
        x_glb_att = self.dropout(self.cross_attn(x_glb2, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)[0])
        x_glb_att = x_glb_att.view_as(x_glb)
        x_glb = self.norm2(x_glb + x_glb_att)

        y = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(1,2))))
        y = self.dropout(self.conv2(y).transpose(1,2))
        return self.norm3(x + y)


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.proj = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for lyr in self.layers:
            x = lyr(x, cross, x_mask, cross_mask, tau, delta)
        if self.norm: x = self.norm(x)
        if self.proj: x = self.proj(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.patch_num = configs.seq_len // configs.patch_len
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
        self.use_norm = configs.use_norm
        self.camera = configs.camera

        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)
        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

        layers = [
            EncoderLayer(
                AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                               configs.d_model, configs.n_heads),
                AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                               configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff, configs.dropout, configs.activation
            ) for _ in range(configs.e_layers)
        ]
        self.encoder = Encoder(layers, norm_layer=nn.LayerNorm(configs.d_model))

        head_nf = configs.d_model * (self.patch_num + 1)
        num_classes = 6 if self.camera else 1
        self.head = FlattenHead(self.n_vars, head_nf, configs.pred_len, num_classes, configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        if self.use_norm:
            mean = x_enc.mean(1, keepdim=True).detach()
            std = torch.sqrt(torch.var(x_enc,1,keepdim=True,unbiased=False)+1e-5)
            x_enc = (x_enc - mean) / std

        bs = x_enc.size(0)
        en, n_vars = self.en_embedding(x_enc[:, :, -1:].permute(0,2,1))
        cross = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)
        enc_out = self.encoder(en, cross)
        enc_out = enc_out.view(bs, n_vars, enc_out.size(-2), enc_out.size(-1)).permute(0,1,3,2)

        dec_out = self.head(enc_out)  # [bs, pred_len, C]
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            raise NotImplementedError
