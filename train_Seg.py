import os
import time
import torch
import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data_process import get_dataset2
from models.AutoEncoder import UnifiedAutoencoder
from models.TopologySeg import NeuralIterativeAP


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_memory(mem_bytes):
    """Convert bytes to MB."""
    return mem_bytes / (1024 ** 2)


@torch.no_grad()
def collect_active_prototypes(ae, loader, dataset_name, device):
    """
    Collect ONLY activated prototypes for a dataset
    Returns:
        active_proto_emb: (K_active, D)
        proto_usage:      (K_active,)  sum=1
    """
    proto_ids = []

    for x, _ in loader:
        out = ae(x.float().to(device), dataset_name=dataset_name)
        proto_ids.append(out["prototype_id"].detach())

    proto_ids = torch.cat(proto_ids)   # (N,)
    proto_ids = proto_ids.cpu()

    # ---------- count usage ----------
    uniq_ids, counts = torch.unique(proto_ids, return_counts=True)
    usage = counts.float()
    usage = usage / usage.sum()        # normalize

    # ---------- get embeddings from codebook ----------
    codebook = ae.topology.codebook.detach().cpu()  # (K_total, D)
    active_proto_emb = codebook[uniq_ids]           # (K_active, D)

    return active_proto_emb, usage


def train_one_dataset(
    ap_net,
    ae,
    loader,
    dataset_name,
    optimizer,
    device,
    target_ratio=0.4,
    lambda_pref=1.0,
    lambda_use=1.0,
    lambda_div=0.1
):
    ap_net.train()
    ae.eval()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()

    # -------- collect active prototypes --------
    proto_emb, proto_usage = collect_active_prototypes(
        ae, loader, dataset_name, device
    )

    proto_emb = proto_emb.to(device)          # (K, D)
    proto_usage = proto_usage.to(device)      # (K,)
    K = proto_emb.size(0)

    if K <= 1:
        return {
            "loss": 0.0,
            "K": K,
            "time_sec": 0.0,
            "gpu_memory_mb": 0.0
        }

    out = ap_net(proto_emb, proto_usage)

    R = out["responsibility"]        # (K, K)
    exemplar_prob = out["exemplar_prob"]  # (K,)

    # ======================================================
    # 1. Exemplar count control (soft but strong)
    # ======================================================
    target_k = max(2, int(target_ratio * K))
    loss_pref = ((exemplar_prob.sum() - target_k) / K) ** 2

    # ===================================E===================
    # 2. Exemplar must be USED
    # each exemplar should attract some responsibility
    # ======================================================
    # how much mass flows INTO each prototype
    proto_inflow = R.sum(dim=0) / K          # (K,)
    loss_use = F.mse_loss(
        proto_inflow,
        exemplar_prob.detach()
    )

    # ======================================================
    # 3. Exemplar diversity
    # ======================================================
    emb_norm = F.normalize(proto_emb, dim=1)
    sim = emb_norm @ emb_norm.T               # (K, K)

    mask = exemplar_prob.unsqueeze(0) * exemplar_prob.unsqueeze(1)
    loss_div = (sim * mask).sum() / (mask.sum() + 1e-6)

    # ======================================================
    # Total loss
    # ======================================================
    loss = (
        lambda_pref * loss_pref +
        lambda_use * loss_use +
        lambda_div * loss_div
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Runtime Statistics
    elapsed_time = time.time() - start_time
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated(device)
        peak_mem_mb = format_memory(peak_mem)
    else:
        peak_mem_mb = 0.0

    return {
        "loss": loss.item(),
        "K": K,
        "time_sec": elapsed_time,
        "gpu_memory_mb": peak_mem_mb
    }


def train(
    ae,
    ap_net,
    datasets,
    device,
    epochs=100,
    lr=1e-4,
    batch_size=512,
    save_dir="./Result"
):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(ap_net.parameters(), lr=lr)

    best_loss = float("inf")

    total_params = count_parameters(ap_net)
    print("\n================================================")
    print("Attention-Based Exemplar Discovery Statistics")
    print("================================================")
    print(f"Trainable Parameters: {total_params:,}")

    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(device)}")

    print("================================================\n")

    total_training_start = time.time()

    for epoch in range(1, epochs + 1):
        losses = []
        Ks = []

        epoch_time_acc = []
        epoch_mem_acc = []

        for name, dataset in datasets:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            stats = train_one_dataset(
                ap_net=ap_net,
                ae=ae,
                loader=loader,
                dataset_name=name,
                optimizer=optimizer,
                device=device,
                target_ratio=0.2
            )

            if stats["K"] > 1:
                losses.append(stats["loss"])
                Ks.append(stats["K"])

                epoch_time_acc.append(stats["time_sec"])
                epoch_mem_acc.append(stats["gpu_memory_mb"])

                print(
                    f"[{name}] "
                    f"loss={stats['loss']:.4f} | "
                    f"active_proto={stats['K']} | "
                    f"time={stats['time_sec']:.2f}s | "
                    f"GPU={stats['gpu_memory_mb']:.2f} MB"
                )

        if len(losses) == 0:
            continue

        avg_loss = sum(losses) / len(losses)
        avg_K = sum(Ks) / len(Ks)

        avg_time = sum(epoch_time_acc) / len(epoch_time_acc)
        avg_mem = sum(epoch_mem_acc) / len(epoch_mem_acc)

        print(
            f"\n[Epoch {epoch:03d}] "
            f"loss={avg_loss:.4f} | "
            f"active_proto≈{avg_K:.1f} | "
            f"avg_time={avg_time:.2f}s | "
            f"avg_gpu={avg_mem:.2f} MB | "
            f"pref={ap_net.preference.item():.3f} | "
            f"tau={ap_net.tau:.3f}"
        )

        # ---- anneal exemplar hardening ----
        if epoch % 5 == 0:
            ap_net.anneal_temperature(0.9)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                ap_net.state_dict(),
                os.path.join(save_dir, "neural_iterative_ap.pt")
            )
            print("✓ Saved best NeuralIterativeAP")

    total_training_time = time.time() - total_training_start
    print("\n================================================")
    print("Final Training Statistics")
    print("================================================")
    print(f"Total Training Time: {total_training_time:.2f} s")

    if torch.cuda.is_available():
        final_peak_mem = torch.cuda.max_memory_allocated(device)
        print(
            f"Peak GPU Memory Usage: "
            f"{format_memory(final_peak_mem):.2f} MB"
        )

    print(f"Model Parameters: {total_params:,}")
    print("================================================")


# ============================================================
# Main Training
# ============================================================
def main(args):
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # ---------- AE ----------
    ae = UnifiedAutoencoder(
        emb_dim=args.emb_dim,
        prototypes=args.prototypes,
        enc_layers=args.enc_layers,
        enc_heads=args.enc_heads,
        dropout=args.dropout,
        max_len=args.max_len
    )

    for dp, fp, name in zip(args.data_paths, args.fea_paths, args.names):
        feature_list, _ = get_dataset2(dp, fp)
        if name not in ae.encoder.adapters:
            ae.register_dataset_adapter(name, input_dim=1, out_dim=len(feature_list))

    ckpt = torch.load(args.ae_ckpt_path, map_location=device)
    ae.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt, strict=False)
    ae.to(device).eval()

    # ---------- Segmentation Net ----------
    ap_net = NeuralIterativeAP(
        dim=args.emb_dim,
        iters=args.iters,
        heads=args.heads,
    ).to(device)

    # ---------- Load datasets ----------
    datasets = []
    for dp, fp, name in zip(args.data_paths, args.fea_paths, args.names):
        _, dataset = get_dataset2(dp, fp)
        datasets.append((name, dataset))

    train(
        ae,
        ap_net,
        datasets,
        device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae_ckpt_path", type=str, default="./Result/best_model.pt")
    parser.add_argument("--data_paths", nargs="+", default=[
        "../datas/breast_cancer_wisconsin_original/test.csv", "../datas/credit_approval/test.csv", "../datas/dermatology/test.csv",
        "../datas/hayes_roth/test.csv", "../datas/heart_disease/test.csv", "../datas/house_votes/test.csv",
        "../datas/lenses/test.csv", "../datas/lung_cancer/test.csv",  "../datas/mammographic_mass/test.csv",
        "../datas/primary_tumor/test.csv", "../datas/promoter_sequence/test.csv", "../datas/solar_flare/test.csv",
        "../datas/soybean_small/test.csv", "../datas/tic+tac+toe+endgame/test.csv", "../datas/zoo/test.csv"
    ])
    parser.add_argument("--fea_paths", nargs="+", default=[
        "../datas/breast_cancer_wisconsin_original/list.txt", "../datas/credit_approval/list.txt", "../datas/dermatology/list.txt",
        "../datas/hayes_roth/list.txt", "../datas/heart_disease/list.txt", "../datas/house_votes/list.txt",
        "../datas/lenses/list.txt", "../datas/lung_cancer/list.txt",  "../datas/mammographic_mass/list.txt",
        "../datas/primary_tumor/list.txt", "../datas/promoter_sequence/list.txt", "../datas/solar_flare/list.txt",
        "../datas/soybean_small/list.txt", "../datas/tic+tac+toe+endgame/list.txt", "../datas/zoo/list.txt"
    ])
    parser.add_argument("--names", nargs="+", default=[
        "breast_cancer_wisconsin_original", "credit_approval", "dermatology",
        "hayes_roth", "heart_disease", "house_votes",
        "lenses", "lung_cancer",  "mammographic_mass",
        "primary_tumor", "promoter_sequence", "solar_flare",
        "soybean_small", "tic+tac+toe+endgame", "zoo"
    ])

    # AE args (must match AE used during pretrain)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--prototypes", type=int, default=64)
    parser.add_argument("--enc_layers", type=int, default=2)
    parser.add_argument("--enc_heads", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)

    parser.add_argument("--save_dir", type=str, default="./Result")
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    main(args)

