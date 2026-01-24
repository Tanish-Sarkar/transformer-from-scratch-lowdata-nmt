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


optimizers = optim.Adam(
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