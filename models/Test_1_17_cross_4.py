# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Embed import DataEmbedding, DataEmbedding_inverted
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FourierCorrelation_Test22 import FourierBlock
from layers.Autoformer_EncDec_TestV3 import Encoder, EncoderLayer, my_Layernorm, series_decomp
from layers.SelfAttention_Family import FullAttention
from math import sqrt
import numpy as np
from scipy.signal import welch, csd


class PhaseBiasedCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_h = n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.log_omega = nn.Parameter(torch.randn(n_heads, 1, 1) - 2)

    def forward(self, Q_in: torch.Tensor, K_in: torch.Tensor, V_in: torch.Tensor, attn_mask=None):
        B, L_q, _ = Q_in.shape
        _, L_k, _ = K_in.shape

        Q = self.q_proj(Q_in).view(B, L_q, self.n_h, self.d_k).transpose(1, 2)
        K = self.k_proj(K_in).view(B, L_k, self.n_h, self.d_k).transpose(1, 2)
        V = self.v_proj(V_in).view(B, L_k, self.n_h, self.d_k).transpose(1, 2)
        logits = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        idx_q = torch.arange(L_q, device=Q_in.device, dtype=torch.float)
        idx_k = torch.arange(L_k, device=Q_in.device, dtype=torch.float)
        delta_t = idx_q[:, None] - idx_k[None, :]  # (L_q, L_k)

        omega = torch.exp(self.log_omega)  # (H, 1, 1)
        phase_diff = 2 * math.pi * omega * delta_t.unsqueeze(0)  # (H, L_q, L_k)
        phase_bias = torch.cos(phase_diff)

        logits = logits + phase_bias.unsqueeze(0)

        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            logits = logits.masked_fill(attn_mask == 0, -1e30)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, L_q, -1)
        return self.o_proj(out), attn

class SoftMaskedCrossAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, mask_fn, dropout: float = 0.1):
        super().__init__()
        self.d_k   = d_model // n_heads
        self.n_h   = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.logit_alpha = nn.Parameter(torch.zeros(n_heads))
        self.mask_fn = mask_fn

    def forward(self, Q_in: torch.Tensor, K_in: torch.Tensor, V_in: torch.Tensor, attn_mask=None):

        B, L_q, _ = Q_in.shape
        _, L_k, _ = K_in.shape

        Q = self.q_proj(Q_in).view(B, L_q, self.n_h, self.d_k).transpose(1,2)  # (B,H,L_q,d_k)
        K = self.k_proj(K_in).view(B, L_k, self.n_h, self.d_k).transpose(1,2)  # (B,H,L_k,d_k)
        V = self.v_proj(V_in).view(B, L_k, self.n_h, self.d_k).transpose(1,2)  # (B,H,L_k,d_k)

        logits = (Q @ K.transpose(-2,-1)) / sqrt(self.d_k)  # (B,H,L_q,L_k)

        M = self.mask_fn(
            n_heads_call=self.n_h,
            L_q=L_q,
            L_k=L_k,
            device=logits.device,
            q_seq=Q_in,
            k_seq=K_in
        )                               # (H,L_q,L_k)
        M = M.unsqueeze(0)              # (1,H,L_q,L_k)

        alpha = torch.sigmoid(self.logit_alpha).view(1, self.n_h, 1, 1)   # (1,H,1,1)

        if attn_mask is not None:
            if attn_mask.dim() == 3: # (B, L_q, L_k)
                attn_mask_expanded = attn_mask.unsqueeze(1) # -> (B, 1, L_q, L_k)
            elif attn_mask.dim() == 4 and attn_mask.shape[1] == 1: # (B, 1, L_q, L_k)
                attn_mask_expanded = attn_mask
            elif attn_mask.dim() == 4 and attn_mask.shape[1] == self.n_h: # (B, H, L_q, L_k)
                attn_mask_expanded = attn_mask
            else:
                 raise ValueError(f"Unsupported attn_mask shape: {attn_mask.shape} for logits shape {logits.shape}")
            if attn_mask_expanded.dtype == torch.bool:
                logits = logits.masked_fill(~attn_mask_expanded, -1e30)
            else:
                logits = logits.masked_fill(attn_mask_expanded == 0, -1e30)


        attn = F.softmax(logits, dim=-1)
        gate = alpha * M + (1 - alpha)            # (1,H,L_q,L_k)
        attn = attn * gate                        # Value-gating
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1,2).contiguous().view(B, L_q, -1)
        if self.training and (torch.rand(()) < 0.001):
            print(f"[MASK-DBG] α_mean={alpha.mean().item():.3f} (min/max: {alpha.min().item():.3f}/{alpha.max().item():.3f}), "
                   f"gate.mean={gate.mean().item():.3f}, gate.std={gate.std().item():.3f}, "
                   f"M.std={M.std().item():.3f}")
        return self.o_proj(out), attn
    
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    
class DampedBiAttention(nn.Module):

    def __init__(self, d_model: int, nhead: int, *, lambda_init: float = 0.1,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.lambda_fwd = nn.Parameter(torch.full((nhead, 1, 1), lambda_init))
        self.lambda_bwd = nn.Parameter(torch.full((nhead, 1, 1), lambda_init))
        self.dropout = nn.Dropout(dropout)
        self.dt = 1.0

    # ---------------------------------------------------------------------
    def forward(self, x, attn_mask):

        B, T, _ = x.shape
        H = self.nhead
        d_head = self.d_model // H

        # project to Q,K,V and split heads
        qkv = self.qkv_proj(x)  # (B, T, 3d)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, H, d_head).transpose(1, 2)  # (B, H, T, d_head)
        k = k.view(B, T, H, d_head).transpose(1, 2)
        v = v.view(B, T, H, d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)
        idx = torch.arange(T, device=x.device)
        rel = idx[None, :] - idx[:, None]           
        lam_fwd = F.softplus(self.lambda_fwd)       
        lam_bwd = F.softplus(self.lambda_bwd)
        decay = torch.where(
            rel < 0, -lam_bwd * self.dt * rel.abs().float(),   
            torch.where(
                rel > 0, -lam_fwd * self.dt * rel.float(),     
                torch.zeros_like(rel, dtype=torch.float)       
            )
        )
        scores = scores + decay.unsqueeze(0)

        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, -1e30)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        y = torch.matmul(attn, v)  # (B, H, T, d_head)
        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(y)
        # print("DampedBiAttention attn",attn.shape)
        return out, attn

class Decoder(nn.Module):

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        
    def forward(self, x, cross_q1, cross_q2, x_mask=None, cross_mask=None, trend=None):
        # trend shape: [B, L, 4]
        residual_trend = None
        for layer in self.layers:
            x, residual_trend = layer(x, cross_q1, cross_q2, x_mask=x_mask, cross_mask=cross_mask)
        
        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        
        seasonal_part = x
        if trend.shape[1] != residual_trend.shape[1]:
            trend = trend[:, -residual_trend.shape[1]:, :]
        trend_part = trend + residual_trend
        return seasonal_part, trend_part
           
class DecoderLayer(nn.Module):
    
    def __init__(self, self_attention, cross_attention_fwd, cross_attention_rev, d_model, d_inner, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        # forward:  Q = external → internal
        self.cross_attention_fwd = cross_attention_fwd
        # reverse:  Q = internal → enriched external
        self.cross_attention_rev = cross_attention_rev
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        
        self.norm3 = my_Layernorm(d_model)
        self.norm_attn_fuse_output = my_Layernorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
        self.fusion_projection = nn.Linear(d_model * 2, d_model)
        self.trend_projection = nn.Linear(d_model, d_inner)

        self.global_q = nn.Parameter(torch.randn(1, 1, d_model))

        self.ctx_pool_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            batch_first=True,
        )


    def forward(self, x_sa_input, x_ca_q1_input, x_ca_q2_input, x_mask=None, cross_mask=None):
        
        B, L, _ = x_sa_input.shape

        sa_attn_out, attn = self.self_attention(x_sa_input, attn_mask=x_mask)
        self.attn = attn
        ca2_attn_out, ca2_attn = self.cross_attention_fwd(x_ca_q2_input, x_ca_q1_input, x_ca_q1_input, attn_mask=cross_mask)
        enriched_x_ca_q2 = x_ca_q2_input + self.dropout(ca2_attn_out)
        ca1_attn_out, ca1_attn = self.cross_attention_rev(x_ca_q1_input, enriched_x_ca_q2, enriched_x_ca_q2, attn_mask=cross_mask)
        self.cross_attn_fwd = ca2_attn
        self.cross_attn_rev = ca1_attn
        
        global_q = self.global_q.repeat(B, 1, 1)                         # [B, 1, D]
        summary, _ = self.ctx_pool_attn(global_q, ca1_attn_out, ca1_attn_out)
        summary = summary.expand(-1, L, -1)                              # broadcast → [B, L, D]
        fused_features = torch.cat((sa_attn_out, summary), dim=-1) 
        combined_attn_out = self.fusion_projection(fused_features) 
        
        x = x_sa_input + self.dropout(combined_attn_out) 
        x = self.norm_attn_fuse_output(x)
        
        x, trend1 = self.decomp1(x)

        residual_ffn = x 
        x_norm3 = self.norm3(x)
        x_ffn_processed = x_norm3.transpose(-1, 1)
        x_ffn_processed = self.dropout(self.activation(self.conv1(x_ffn_processed)))
        x_ffn_processed = self.dropout(self.conv2(x_ffn_processed))
        x_ffn_processed = x_ffn_processed.transpose(-1, 1)
        
        x = residual_ffn + x_ffn_processed
        
        x, trend2 = self.decomp2(x)
        
        combined_residual_trend = trend1 + trend2
        projected_residual_trend = self.trend_projection(combined_residual_trend)
        
        return x, projected_residual_trend

class Model(nn.Module):


    def __init__(self, configs, version='fourier', mode_select='low', modes=32):

        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        # Decomp
        self.decomp = series_decomp(configs.moving_avg)
        
        # Encoder embedding
        self.en_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.ca_ex_embedding = DataEmbedding(12, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.ca_en_embedding = DataEmbedding(4, configs.d_model, configs.embed, configs.freq, configs.dropout)

        encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                        out_channels=configs.d_model,
                                        seq_len=self.seq_len,
                                        modes=self.modes,
                                        mode_select_method=self.mode_select)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    self_attention=DampedBiAttention(
                        d_model=configs.d_model,
                        nhead=configs.n_heads,
                        dropout=configs.dropout
                    ),
                    cross_attention_fwd=PhaseBiasedCrossAttention(
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                        dropout=configs.dropout
                    ),
                    cross_attention_rev=PhaseBiasedCrossAttention(
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                        dropout=configs.dropout
                    ),
                    d_model=configs.d_model,
                    d_inner=4,
                    d_ff=configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, 4, bias=True)
        )

        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        B, L_enc, N_total = x_enc.shape
        external_vars_batch_torch = x_enc[:, :, :12]
        internal_vars_batch_torch = x_enc[:, :, -4:]

        en_embed = self.en_embedding(x_enc, x_mark_enc)

        mean = torch.mean(internal_vars_batch_torch, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        _, trend_init_from_enc = self.decomp(internal_vars_batch_torch)
        trend_for_decoder = torch.cat([trend_init_from_enc[:, -self.label_len:, :], mean], dim=1)

        en_encoder, attns = self.encoder(en_embed, attn_mask=None)         
        
        dec_input_ca_q1 = self.ca_en_embedding(internal_vars_batch_torch, None)
        dec_input_ca_q2 = self.ca_ex_embedding(external_vars_batch_torch, None)

        seasonal_part, trend_part = self.decoder(en_encoder, 
                                                 dec_input_ca_q1, 
                                                 dec_input_ca_q2, 
                                                 trend=trend_for_decoder) 
    
        dec_out = seasonal_part + trend_part
            
        return dec_out
        

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # enc
        enc_out = self.en_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.en_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.en_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
            or self.task_name == "forecasting"
            or self.task_name == "forecasting_frequency_loss"
            or self.task_name == "forecasting_loss_finetune"
        ):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
