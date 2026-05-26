import os
import time
import argparse
import logging
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.data_process import get_dataset, get_feature_list
from models.AutoEncoder import UnifiedAutoencoder

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
transformers_logger = logging.getLogger('transformer')
transformers_logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="You are resizing the embedding layer without providing a pad_to_multiple_of parameter.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_memory(mem_bytes):
    """Convert bytes to MB."""
    return mem_bytes / (1024 ** 2)
# ------ Augmentations for contrastive view ----------
def augment_batch(x, feat_noise=0.05, feature_mask_prob=0.1):
    """ Simple augmentation for tabular features: - additive gaussian noise
    - feature dropout (randomly zero some features per sample)
    x: (B, N) returns same shape """
    x2 = x + feat_noise * torch.randn_like(x)
    if feature_mask_prob > 0:
        mask = (torch.randn_like(x2) > feature_mask_prob).float()
        x2 = x2 * mask
    return x2

# --- Training helpers ---
def train_one_epoch_on_dataset(
        model,
        optimizer,
        loader,
        device,
        dataset_name,
        lambda_contrast,
        lambda_entropy,
        feat_noise,
        mask_prob
):
    """ Train model for one epoch on given dataset loader. Returns average losses dict. """
    model.train()
    total_loss_acc = 0.0
    recon_acc = 0.0
    commit_acc = 0.0
    nce_acc = 0.0
    entropy_acc = 0.0
    count = 0
    # accumulate usage counts per epoch for entropy regularizer optionally
    all_indices = []

    #------- GPU Memory Monitor --------------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()

    for batch in loader:
        x = batch.to(device)
        B = x.shape[0]

        out = model(x, dataset_name=dataset_name)
        sample_emb = out['sample_emb'] # pooled embedding
        indices = out['prototype_id'] # (B,) tensor
        recon_loss = out.get('recon_loss', None)
        commit_loss = out.get('commit_loss', None)
        base_total = out.get('total_loss', None)

        # augmentation and pooled embedding for augmented view
        x_aug = augment_batch(x, feat_noise=feat_noise, feature_mask_prob=mask_prob)
        enc_aug, pooled_aug = model.encoder(x_aug, dataset_name=dataset_name) # pooled_aug (B,D)

        # contrastive loss between sample_emb and pooled_aug
        # use model.compute_contrastive_loss
        nce_loss = model.compute_contrastive_loss(sample_emb, pooled_aug)

        # codebook entropy loss computed from indices
        # model.codebook_entropy_loss works on indices
        entropy_loss = model.codebook_entropy_loss(indices)
        total_loss = base_total + lambda_contrast * nce_loss + lambda_entropy * entropy_loss

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # accumulate metrics
        total_loss_acc += total_loss.item()
        if recon_loss is not None:
            recon_acc += recon_loss.item()

        if commit_loss is not None:
            commit_acc += commit_loss.item()
        nce_acc += nce_loss.item()
        entropy_acc += entropy_loss.item()
        count += 1
        # accumulate indices for monitoring (not used further here)
        if isinstance(indices, torch.Tensor):
            all_indices.append(indices.detach().cpu())

    epoch_time = time.time() - start_time
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated(device)
        peak_mem_mb = format_memory(peak_mem)
    else:
        peak_mem_mb = 0.0

    # epoch averages
    avg = lambda name, val: (val / count) if count > 0 else 0.0

    stats = {
        'loss': avg('loss', total_loss_acc),
        'recon_loss': avg('recon', recon_acc),
        'commit_loss': avg('commit', commit_acc),
        'nce_loss': avg('nce', nce_acc),
        'entropy_loss': avg('entropy', entropy_acc),
        'batches': count,
        # --------- Resource Statistics --------
        'time_sec': epoch_time,
        'peak_memory_mb': peak_mem_mb
    }
    return stats

# ------- Main multi-round sequential trainer ------
def train_sequential_datasets(
        model,
        dataset_paths,
        features_paths,
        dataset_names,
        device,
        rounds=100,
        batch_size=128,
        lr=4e-4,
        lambda_contrast=0.5,
        lambda_entropy=0.1,
        feat_noise=0.05,
        mask_prob=0.1,
        save_dir="./Result"
):
    for i in range(len(dataset_paths)):
        feature_list = get_feature_list(features_paths[i])
        input_dim = 1
        out_dim = len(feature_list)

        if dataset_names[i] not in model.decoders and dataset_names[i] not in model.encoder.adapters:
            model.register_dataset_adapter(dataset_names[i], input_dim, out_dim)
        else:
            # if any of them exists, still ensure decoder exists
            try:
                if dataset_names[i] not in model.decoders:
                    model.register_dataset_adapter(dataset_names[i], input_dim, out_dim)
            except Exception:
                pass

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Parameter Statistics
    total_params = count_parameters(model)
    print("\n================================================")
    print("Model Statistics")
    print("================================================")
    print(f"Trainable Parameters: {total_params:,}")

    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(device)}")

    print("================================================\n")

    # ------------------------ # 3) training loop # ------------------------
    best_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    for r in range(1, rounds + 1):
        losses = []
        print(f"\n=== Global Round {r}/{rounds} ===")
        for i in range(len(dataset_paths)):
            name = dataset_names[i]
            path = dataset_paths[i]

            # load dataset into DataLoader
            feature_list, dataset = get_dataset(path, features_paths[i])
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # per-dataset one epoch within this round
            stats = train_one_epoch_on_dataset(
                model,
                optimizer,
                loader,
                device,
                dataset_name=name,
                lambda_contrast=lambda_contrast,
                lambda_entropy=lambda_entropy,
                feat_noise=feat_noise,
                mask_prob=mask_prob
            )

            losses.append(stats["loss"])

            print(
                f"[{name}] "
                f"loss={stats['loss']:.4f} | "
                f"time={stats['time_sec']:.2f}s | "
                f"GPU Memory={stats['peak_memory_mb']:.2f} MB"
            )

        avg_loss = sum(losses) / max(len(losses), 1)
        print(f"[Epoch {r}] avg_loss={avg_loss:.4f}")
        # save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"Saved best model to {ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--emb_dim", default=64, type=int, help='Embedding Size')
    parser.add_argument("--enc_layers", default=2, type=int)
    parser.add_argument("--enc_heads", default=4, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--prototypes", default=64, type=int)
    parser.add_argument("--max_len", default=512, type=int)
    parser.add_argument("--batch_size", default=128, type=int, help="Batch Size")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--rounds", type=int, default=100, help="Number of global rounds")

    parser.add_argument("--lambda_contrast", type=float, default=0.5)
    parser.add_argument("--lambda_entropy", type=float, default=0.1)
    parser.add_argument("--feat_noise", type=float, default=0.05)
    parser.add_argument("--mask_prob", type=float, default=0.1)
    parser.add_argument("--save_dir", default="./Result", type=str)

    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.device = torch.device(args.device)

    # prepare datasets
    dataset_paths = [
        "../datas/balance_scale/test.csv", "../datas/breast_cancer_wisconsin_original/test.csv", "../datas/car_evaluation/test.csv",
        "../datas/credit_approval/test.csv", "../datas/dermatology/test.csv", "../datas/hayes_roth/test.csv",
        "../datas/heart_disease/test.csv", "../datas/house_votes/test.csv", "../datas/lenses/test.csv",
        "../datas/lung_cancer/test.csv", "../datas/lymphography/test.csv", "../datas/mammographic_mass/test.csv",
        "../datas/primary_tumor/test.csv", "../datas/promoter_sequence/test.csv", "../datas/solar_flare/test.csv",
        "../datas/soybean_small/test.csv", "../datas/tic+tac+toe+endgame/test.csv", "../datas/zoo/test.csv"
    ]

    feature_paths = [
        "../datas/balance_scale/list.txt", "../datas/breast_cancer_wisconsin_original/list.txt", "../datas/car_evaluation/list.txt",
        "../datas/credit_approval/list.txt", "../datas/dermatology/list.txt", "../datas/hayes_roth/list.txt",
        "../datas/heart_disease/list.txt", "../datas/house_votes/list.txt", "../datas/lenses/list.txt",
        "../datas/lung_cancer/list.txt", "../datas/lymphography/list.txt", "../datas/mammographic_mass/list.txt",
        "../datas/primary_tumor/list.txt", "../datas/promoter_sequence/list.txt", "../datas/solar_flare/list.txt",
        "../datas/soybean_small/list.txt", "../datas/tic+tac+toe+endgame/list.txt", "../datas/zoo/list.txt"
    ]

    dataset_names = [
        "balance_scale", "breast_cancer_wisconsin_original", "car_evaluation",
        "credit_approval", "dermatology", "hayes_roth",
        "heart_disease", "house_votes", "lenses",
        "lung_cancer", "lymphography", "mammographic_mass",
        "primary_tumor", "promoter_sequence", "solar_flare",
        "soybean_small", "tic+tac+toe+endgame", "zoo"
    ]

    model = UnifiedAutoencoder(
        emb_dim=args.emb_dim,
        prototypes=args.prototypes,
        enc_layers=args.enc_layers,
        enc_heads=args.enc_heads,
        dropout=args.dropout,
        max_len=args.max_len
    ).to(args.device)

    train_sequential_datasets(
        model=model,
        dataset_paths=dataset_paths,
        features_paths=feature_paths,
        dataset_names=dataset_names,
        device=args.device,
        rounds=args.rounds,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_contrast=args.lambda_contrast,
        lambda_entropy=args.lambda_entropy,
        feat_noise=args.feat_noise,
        mask_prob=args.mask_prob,
        save_dir=args.save_dir
    )