import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import MultiHeadAttention
import math


class DecoderLayer(nn.Module):
    def __init__(self, d_model,num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        """
        x: (batch, tgt_seq_len, d_model)
        enc_output: (batch, src_seq_len, d_model)
        tgt_mask: (batch, 1, tgt_seq_len, tgt_seq_len)
        src_mask: (batch, 1, 1, src_seq_len)
        """
        # 1. Masked self-attention (causal)
        self_attn, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn))

        # 2.Cross-attention (decoder attends to encoder output)
        cross_attn, _ = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn))

        # 3. Feed Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        max_len=5000,
        dropout=0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = None

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        """
        tgt: (batch, tgt_seq_len)
        """
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)

        return x