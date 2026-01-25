import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch.nn as nn
import math

def get_loss(pad_idx):
    return nn.CrossEntropyLoss(ignore_index=pad_idx)

# This class matters a LOT for stability in low-data.
class TransformerLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (
            self.d_model ** -0.5 *
            min(
                self.step_num ** -0.5,
                self.step_num * self.warmup_steps ** -1.5
            )
        )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
