import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        return output, attention
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0   # Perfect divisibility should be their

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model) 
        self.W_k = nn.Linear(d_model, d_model) 
        self.W_v = nn.Linear(d_model, d_model) 
        self.W_o = nn.Linear(d_model, d_model) 

        self.attention = ScaledDotProductAttention()

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
    
    def combine_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        x, attention = self.attention(Q, K, V, mask)

        x = self.combine_heads(x)
        x = self.W_o(x)
        return x, attention