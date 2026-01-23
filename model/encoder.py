import torch 
import torch.nn as nn
import torch.nn. functional as F
from model.attention import MultiHeadAttention
from model.positional_encoding import PositionalEncoding


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mark=None):
        """
        x: (batch, src_seq_len, d_model)
        src_mask: (batch, 1, 1, src_seq_len)
        """

        # 1. Self-attention
        attn_output,_ = self.self_attention(x,x,x,src_mark)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. Feed Foward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x 
    
class Encoder(nn.Module):
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
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) 
                                     for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, src, src_mask=None):
        """
        src: (batch, src_seq_len)
        """
        x = self.embedding(src) * (self.d_model ** 0.5)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x