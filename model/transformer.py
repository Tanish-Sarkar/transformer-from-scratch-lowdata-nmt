import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from model.positional_encoding import PositionalEncoding

def make_src_mask(src, pad_idx):
    """
    src: (batch, src_len)
    returns: (batch, 1, 1, src_len)
    """
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt, pad_idx):
    """
    tgt: (batch, tgt_len)
    return: (batch, 1, tgt_len, tgt_len)
    """
    batch_size, tgt_len = tgt.size()

    # Padding mask
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    # (batch, 1, 1, tgt_len)

    # Causal mask
    causal_mask = torch.tril(
        torch.ones((tgt_len, tgt_len), device=tgt.device)
    ).bool()
    # (tgt_len, tgt_len)

    return pad_mask & causal_mask



class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        max_len=5000,
        dropout=0.1
    ):
        super().__init__()
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.src_pad_idx = src_pad_idx 
        self.tgt_pad_idx = tgt_pad_idx 



    def forward(self, src, tgt):
        """
        src: (batch, src_len)
        tgt: (batch, tgt_len)
        """
        src_mask = make_src_mask(src, self.src_pad_idx)
        tgt_mask = make_tgt_mask(tgt, self.tgt_pad_idx)

        # Encoder
        enc_output = self.encoder(src, src_mask)
        
        # Decoder
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)

        logits = self.fc_out(dec_output)
        return logits