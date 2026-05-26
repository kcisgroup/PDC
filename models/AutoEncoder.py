import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from vector_quantize_pytorch import VectorQuantize


# ------ Utilities --------
class InputAdapter(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.adapter = nn.Linear(input_dim, emb_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        return self.adapter(x)


class DecoderHead(nn.Module):
    def __init__(self, emb_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, emb):
        return self.net(emb)


# -------- Universal Transformer Encoder --------
class UniversalTransformerEncoder(nn.Module):
    def __init__(self, emb_dim=64, num_layers=2, num_heads=4, dropout=0.1, mlp_ratio=4.0, max_len=512):
        super().__init__()
        self.emb_dim = emb_dim
        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * int(mlp_ratio),
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.global_norm = nn.LayerNorm(emb_dim)
        # pos emb
        self.max_len = max_len
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, emb_dim))
        # storage for adapters (dataset_name -> InputAdapter)
        self.adapters = nn.ModuleDict()

    def register_adapter(self, dataset_name, input_dim):
        if dataset_name in self.adapters:
            pass
        self.adapters[dataset_name] = InputAdapter(input_dim, self.emb_dim)

    def forward(self, x, dataset_name):
        if len(self.adapters) == 0:
            raise RuntimeError("No input adapters registered. Call register_adapter(dataset_name, input_dim).")
        if dataset_name is None:
            if len(self.adapters) == 1:
                adapter = next(iter(self.adapters.values()))
            else:
                raise RuntimeError("Multiple adapters registered; pass dataset_name to forward().")
        else:
            if dataset_name not in self.adapters:
                raise KeyError(f"Adapter for dataset '{dataset_name}' not registered.")
            adapter = self.adapters[dataset_name]

        x_proj = adapter(x)
        B, N, D = x_proj.shape
        if N > self.max_len:
            new_max = max(self.max_len * 2, N)
            new_pos = nn.Parameter(torch.randn(1, new_max, self.pos_emb))
            with torch.no_grad():
                new_pos[:, : self.max_len] = self.pos_emb
            self.pos_emb = new_pos
            self.max_len = new_max

        x_proj = x_proj + self.pos_emb[:, :N, :]

        # normalize (helps cross-dataset)
        x_proj = self.global_norm(x_proj)
        x_proj = self.global_norm(x_proj)
        enc = self.encoder(x_proj)
        pooled = enc.mean(dim=1)
        return enc, pooled


# ------- Alignment / Regularization utilities -------
def info_nce_loss(emb_q, emb_k, temperature=0.1):
    """
    Simple InfoNCE between two views emb_q and emb_k.
    emb_q, emb_k: (B, D) with matching order (same samples under two augmentations)
    return scalar loss
    """
    emb_q = F.normalize(emb_q, dim=-1)
    emb_k = F.normalize(emb_k, dim=-1)
    logits = emb_q @ emb_k.t() / temperature   # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_q = F.cross_entropy(logits, labels)
    loss_k = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_q + loss_k)

def codebook_entropy_loss(indices, codebook_size, epsilon=1e-6):
    """
    Encourage uniform usage of codebook across a batch.
    indices: (B,) long tensor of codebook ids
    return KL or negative entropy loss: lower when usage is uniform.
    """
    binc = torch.bincount(indices.view(-1), minlength=codebook_size).float()
    probs = binc / (binc.sum() + 1e-12)
    # entropy
    ent = -(probs * (probs + epsilon).log()).sum()
    # max entropy = Log(K); use normalize negative entropy as loss (want to maximize entropy -> minimize -ent)
    max_ent = math.log(codebook_size + 1e-12)
    # loss = (max_ent - ent) normalized -> smaller when ent near max_ent
    loss = (max_ent - ent) / (max_ent + 1e-12)
    return loss


# ---------- UnifiedAutoencoder ---------------
class UnifiedAutoencoder(nn.Module):
    def __init__(self, emb_dim=64, prototypes=64, enc_layers=2, enc_heads=4, dropout=0.1, max_len=512, default_decoder_hidden=128):
        super().__init__()

        self.emb_dim = emb_dim
        self.encoder = UniversalTransformerEncoder(
            emb_dim=emb_dim,
            num_layers=enc_layers,
            num_heads=enc_heads,
            dropout=dropout,
            max_len=max_len
        )

        # store per-dataset decoder heads
        self.decoders = nn.ModuleDict()    # dataset_name -> DecoderHead
        # shared codebook
        self.topology = VectorQuantize(
            dim=emb_dim,
            codebook_dim=emb_dim,
            codebook_size=prototypes
        )

        # optional hyperparameters for cross-dataset regularization (user can tune or disable)
        self.register_buffer("global_codebook_usage_count", torch.zeros(prototypes))  # debug/monitor only

    def register_dataset_adapter(self, dataset_name, input_dim, out_dim):
        self.encoder.register_adapter(dataset_name, input_dim)
        self.decoders[dataset_name] = DecoderHead(self.emb_dim, out_dim, hidden=max(default_decoder_hidden := 128, self.emb_dim*2))

    def forward(self, x, dataset_name):
        if dataset_name not in self.decoders:
            raise KeyError(f"Decoder for dataset {dataset_name} not registered. Call register_dataset_adapter().")
        enc, sample_emb = self.encoder(x, dataset_name=dataset_name)
        # quantize pooled embeddings (per-sample)
        quantized_emb, indices, commit_loss = self.topology(sample_emb)
        recon = self.decoders[dataset_name](quantized_emb)

        recon_loss = F.mse_loss(
            recon,
            x
        )

        total_loss = recon_loss + commit_loss

        # update codebook usage count (for monitoring only)
        try:
            codebook = getattr(self.topology, "codebook", None)
            if codebook is not None and isinstance(indices, torch.Tensor):
                binc = torch.bincount(indices.detach().view(-1), minlength=codebook.shape[0]).float()
                self.global_codebook_usage_count = self.global_codebook_usage_count + binc.to(
                    self.global_codebook_usage_count.device)
        except Exception:
            pass

        return {
            "sample_emb": sample_emb,             # encoder pooled
            "quantized_emb": quantized_emb,         #  InfoNCE / segmentation
            "prototype_id": indices,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
            "total_loss": total_loss,
            "codebook": getattr(self.topology, "codebook", None),
        }

    # --------- optional helper losses (call from training loop if desired) -----------
    def compute_contrastive_loss(self, emb_view1, emb_view2, temperature=0.1):
        return info_nce_loss(emb_view1, emb_view2, temperature=temperature)

    def codebook_entropy_loss(self, indices):
        codebook = getattr(self.topology, "codebook", None)
        if codebook is None:
            raise RuntimeError("No codebook accessible on topology.")
        K = codebook.shape[0]
        return codebook_entropy_loss(indices, K)

    def get_codebook(self):
        return getattr(self.topology, "codebook", None)

    def codebook_entropy_loss_ema(
            self,
            indices,
            dataset_name,
            decay=0.99,
            eps=1e-8
    ):
        K = self.topology.codebook.size(0)

        if dataset_name not in self.ema_codebook_usage:
            self.ema_codebook_usage[dataset_name] = torch.zeros(
                K, device=indices.device
            )
            self.ema_initialized[dataset_name] = False

        usage = self.ema_codebook_usage[dataset_name]

        batch_hist = torch.bincount(
            indices.detach(), minlength=K
        ).float()
        batch_hist = batch_hist / (batch_hist.sum() + eps)

        if not self.ema_initialized[dataset_name]:
            usage.copy_(batch_hist)
            self.ema_initialized[dataset_name] = True
        else:
            usage.mul_(decay).add_(batch_hist * (1 - decay))

        probs = usage / (usage.sum() + eps)
        entropy = -(probs * (probs + eps).log()).sum()
        max_entropy = math.log(K + eps)

        return (max_entropy - entropy) / max_entropy



