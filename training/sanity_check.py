import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer import Transformer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SRC_VOCAB_SIZE = 50
TGT_VOCAB_SIZE = 50
PAD_IDX = 0

D_MODEL = 64
NUM_LAYERS = 2
NUM_HEADS = 4
D_FF = 128

BATCH_SIZE = 4
SRC_LEN = 6
TGT_LEN = 6


torch.manual_seed(42)

src = torch.randint(1, SRC_VOCAB_SIZE, (BATCH_SIZE, SRC_LEN)).to(DEVICE)
tgt = torch.randint(1, TGT_VOCAB_SIZE, (BATCH_SIZE, TGT_LEN)).to(DEVICE)

# Teacher forcing
tgt_input = tgt[:, :-1]
tgt_output = tgt[:, 1:]

model = Transformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    src_pad_idx=PAD_IDX,
    tgt_pad_idx=PAD_IDX,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    dropout=0.0  # IMPORTANT: disable dropout for sanity
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Starting sanity check...")

for epoch in range(300):
    model.train()
    optimizer.zero_grad()

    logits = model(src, tgt_input)
    loss = criterion(
        logits.reshape(-1, TGT_VOCAB_SIZE),
        tgt_output.reshape(-1)
    )

    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
