"""Small supervised history encoders; input features stay frozen on disk."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


MODELS = ("pair", "pooled", "gru", "transformer", "pooled-gru", "pooled-transformer")


class HistoryClassifier(nn.Module):
    """Predict the latest event using its pair and observations strictly before it.

    Calendar features are gap/10 and elapsed-years/10. The no-time ablation
    retains sequence order. Pooling controls use position-free MLPs with
    approximately the recurrent/attention block's parameter budget.
    """

    def __init__(self, dimension, model="gru", width=128, dropout=0.1, time_mode="gaps"):
        super().__init__()
        if model not in MODELS or time_mode not in {"gaps", "none"}:
            raise ValueError("Unknown history model or time mode")
        if width < 4 or width % 4 or not 0 <= dropout < 1:
            raise ValueError("width must be a positive multiple of 4; dropout must be in [0,1)")
        self.model, self.width, self.time_mode = model, width, time_mode
        self.visual = nn.Linear(dimension, width)
        self.time = nn.Linear(2, width, bias=False)
        if model == "gru":
            self.temporal = nn.GRU(width, width, batch_first=True)
        elif model == "transformer":
            layer = nn.TransformerEncoderLayer(
                width, 4, width * 2, dropout, batch_first=True, activation="gelu")
            self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            # TransformerEncoder clones its first layer. Initialize layers independently.
            for block in self.temporal.layers:
                for parameter in block.parameters():
                    if parameter.ndim > 1:
                        nn.init.xavier_uniform_(parameter)
        elif model.startswith("pooled-"):
            multiplier = 3 if model == "pooled-gru" else 8
            self.temporal = nn.Sequential(nn.Linear(width, width * multiplier), nn.GELU(),
                                          nn.Linear(width * multiplier, width))
        else:
            self.temporal = nn.Identity()
        inputs = width * (4 if model == "pair" else 5) + 1
        self.head = nn.Sequential(nn.Linear(inputs, width), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(width, 1))

    def encode(self, past, time_features, lengths):
        """Each row is an independent past-only prefix, with right padding."""
        tokens = self.visual(past)
        if self.time_mode == "gaps":
            tokens = tokens + self.time(time_features / 10.0)
        positions = torch.arange(past.shape[1], device=past.device)
        valid = positions[None, :] < lengths[:, None]
        if self.model == "gru":
            packed = pack_padded_sequence(tokens, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.temporal(packed)
            return hidden[-1]
        if self.model == "transformer":
            frequency = torch.exp(torch.arange(0, self.width, 2, device=past.device)
                                  * (-math.log(10000.0) / self.width))
            angles = positions[:, None] * frequency[None, :]
            positional = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)
            # No current or future observations occur in this tensor. A causal
            # attention mask is unnecessary inside an already historical prefix.
            encoded = self.temporal(tokens + positional, src_key_padding_mask=~valid)
            return encoded[torch.arange(len(lengths), device=past.device), lengths - 1]
        pooled = (tokens * valid.unsqueeze(-1)).sum(1) / lengths[:, None]
        return self.temporal(pooled)

    def forward(self, past, time_features, lengths, current, pair_gap):
        batch = torch.arange(len(lengths), device=past.device)
        previous = self.visual(past[batch, lengths - 1])
        current = self.visual(current)
        difference = current - previous
        parts = [previous, current, difference, difference.abs()]
        history = self.encode(past, time_features, lengths) if self.model != "pair" else previous
        if self.model != "pair":
            parts.append(history)
        gap = pair_gap[:, None] / 10.0 if self.time_mode == "gaps" else pair_gap[:, None] * 0
        logits = self.head(torch.cat([*parts, gap], dim=1)).squeeze(-1)
        return logits, history


def batch_samples(samples, device, shuffle_past=False, seed=42):
    """Pad past-only prefixes; permute older appearances without moving latest pair."""
    lengths = [len(row["past"]) for row in samples]
    count, dimension = max(lengths), samples[0]["current"].shape[0]
    past = torch.zeros((len(samples), count, dimension), dtype=torch.float32)
    times = torch.zeros((len(samples), count, 2), dtype=torch.float32)
    for i, row in enumerate(samples):
        values = torch.as_tensor(row["past"].copy(), dtype=torch.float32)
        if shuffle_past and lengths[i] >= 3:
            import hashlib
            key = int(hashlib.sha256(f"{seed}:{row['event_id']}".encode()).hexdigest()[:15], 16)
            generator = torch.Generator().manual_seed(key)
            order = torch.randperm(lengths[i] - 1, generator=generator)
            if torch.equal(order, torch.arange(lengths[i] - 1)):
                order = order.roll(1)
            values[:-1] = values[:-1][order]
        past[i, :lengths[i]] = values
        times[i, :lengths[i]] = torch.as_tensor(row["time"], dtype=torch.float32)
    return {"past": past.to(device), "time_features": times.to(device),
            "lengths": torch.tensor(lengths, device=device),
            "current": torch.stack([torch.as_tensor(row["current"]) for row in samples]).to(device),
            "pair_gap": torch.tensor([row["pair_gap"] for row in samples], device=device, dtype=torch.float32)}
