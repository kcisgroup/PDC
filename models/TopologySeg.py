import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralIterativeAP(nn.Module):
    """
    Neural Iterative Affinity Propagation
    - Responsibility: Multi-head attention
    - Availability: neural + usage prior + preference
    - Preference controls number of exemplars
    """
    def __init__(
        self,
        dim,
        heads=4,
        iters=3,
        dropout=0.1,
        usage_coef=1.0,
        pref_init=-1.0,
        pref_learnable=True,
        init_tau=1.0,
        min_tau=0.05
    ):
        super().__init__()

        self.iters = iters
        self.usage_coef = usage_coef

        # temperature for exemplar hardening
        self.tau = init_tau
        self.min_tau = min_tau

        # Attention = responsibility generator
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        # Availability MLP (neural)
        self.avail_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

        # preference bias (core)
        if pref_learnable:
            self.preference = nn.Parameter(torch.full((1,), pref_init))
        else:
            self.register_buffer("preference", torch.tensor(pref_init))

        # prototype refinement
        self.refine = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU()
        )

    def anneal_temperature(self, factor=0.95):
        self.tau = max(self.min_tau, self.tau * factor)

    def forward(self, prototypes, proto_usage):
        """
        prototypes: (K, D)
        proto_usage: (K,) normalized histogram (sum=1)

        Returns:
            responsibility: (K, K)
            availability:   (K,)
            exemplar_prob:  (K,)
            embeddings:  (K, D)
        """
        K, D = prototypes.shape
        h = prototypes

        log_usage = torch.log(proto_usage + 1e-8)

        responsibility = None
        availability = None

        for _ in range(self.iters):
            # ---------------------------
            # Responsibility (Attention)
            # ---------------------------
            # each proto attends to all protos
            attn_out, attn_weight = self.attn(
                h.unsqueeze(0),
                h.unsqueeze(0),
                h.unsqueeze(0)
            )
            responsibility = attn_weight.squeeze(0)  # (K, K)

            # ---------------------------
            # Availability = neural + usage + preference
            # ---------------------------
            A_logit = self.avail_mlp(h).squeeze(-1)  # (K,)
            availability = (
                    A_logit
                    + self.usage_coef * log_usage
                    + self.preference
            )

            # ---------------------------
            # Exemplar probability (annealed)
            # ---------------------------
            exemplar_prob = torch.sigmoid(availability / self.tau)

            # ---------------------------
            # Prototype refinement
            # ---------------------------
            h = responsibility @ h
            h = self.refine(h)

        return {
            "responsibility": responsibility,   # (K, K)
            "availability": availability,       # (K,)
            "exemplar_prob": exemplar_prob,      # (K,)
            "embeddings": h                      # (K, D)
        }
