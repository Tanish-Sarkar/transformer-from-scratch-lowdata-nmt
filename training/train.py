import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch
import torch.optim as optim
from tqdm import tqdm
from model.transformer import Transformer
from data.dataloader import get_dataloaders
from training.loss import get_loss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 2048
DROPOUT = 0.1

BATCH_SIZE = 32
EPOCHS = 10
WARMUP_STEPS = 4000


train_loader, valid_loader, _, src_vocab, tgt_vocab = get_dataloaders(
    batch_size=BATCH_SIZE,
    device=DEVICE
)

SRC_PAD_IDX = src_vocab["<pad>"]
TGT_PAD_IDX = tgt_vocab["<pad>"]

model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    src_pad_idx=SRC_PAD_IDX,
    tgt_pad_idx=TGT_PAD_IDX,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    dropout=DROPOUT
).to(DEVICE)


optimizer = optim.Adam(
    model.parameters(),
    betas=(0.9, 0.98),
    eps=1e-9
)

scheduler = TransformerLRScheduler(
    optimizer,
    d_model=D_MODEL,
    warmup_steps=WARMUP_STEPS
)
criterion = get_loss(TGT_PAD_IDX)

def train_one_epoch(model, dataloader):
    model.train()
    total_loss = 0 
    
    for src, tgt in tqdm(dataloader):
        optimizer.zero_grad()

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, :-1]

        logits = model(src, tgt_input)
        loss = criterion(logits.reshpae(-1, logits.size(-1)), tgt_output.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader) 


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src,tgt in dataloader:
            tgt_input = tgt[:,:-1]
            tgt_output = tgt[:,1:]

            logits = model(src, tgt_input)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            ) 

            total_loss += loss.item()

    return total_loss / len(dataloader)


for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = evaluate(model, valid_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.3f} | "
        f"Val Loss: {val_loss:.3f}"
    )
