import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from collections import Counter
from torch.utils.data import DataLoader
from sklearn.metrics.cluster import pair_confusion_matrix

from utils.data_process import get_dataset2
from models.AutoEncoder import UnifiedAutoencoder
from models.TopologySeg import NeuralIterativeAP


# ============================================================
# Step 1. Extract ACTIVE prototypes + sample info
# ============================================================
@torch.no_grad()
def extract_active_protos(ae, loader, dataset_name, device):
    proto_ids = []
    labels_all = []

    for x, y in loader:
        out = ae(x.float().to(device), dataset_name=dataset_name)
        proto_ids.append(out["prototype_id"].detach().cpu())
        labels_all.append(y.cpu())

    proto_ids = torch.cat(proto_ids)      # (N,)
    labels_all = torch.cat(labels_all)    # (N,)

    # -------- active prototype ids --------
    active_proto_ids, counts = torch.unique(proto_ids, return_counts=True)
    usage = counts.float()
    usage = usage / usage.sum()

    # -------- embeddings from codebook --------
    codebook = ae.topology.codebook.detach().cpu()
    proto_emb = codebook[active_proto_ids]    # (K_active, D)

    return {
        "proto_ids": proto_ids,                 # (N,)
        "labels": labels_all,                   # (N,)
        "active_proto_ids": active_proto_ids,   # (K_active,)
        "proto_emb": proto_emb,                 # (K_active, D)
        "proto_usage": usage                    # (K_active,)
    }

# ============================================================
# Step 2. Prototype → Exemplar → Cluster
# ============================================================
@torch.no_grad()
def cluster_by_exemplar(
    proto_emb,
    exemplar_prob,
    proto_ids,
    active_proto_ids,
    labels,
    exemplar_thresh=0.3,
    min_proto_for_ap=6
):
    """
    Returns:
        sample_pred_labels: (N,)
        num_clusters: int
    """
    device = proto_emb.device
    K = proto_emb.size(0)

    # =====================================================
    # Case 1: prototype number too small → no AP
    # =====================================================
    if K <= min_proto_for_ap:
        # each prototype is a cluster
        protoid_to_cluster = {
            pid.item(): i for i, pid in enumerate(active_proto_ids)
        }
        sample_cluster = torch.zeros_like(labels)
        for i, pid in enumerate(proto_ids.tolist()):
            sample_cluster[i] = protoid_to_cluster[pid]

        # majority vote per cluster
        cluster_majority = {}
        for c in range(K):
            idx = (sample_cluster == c).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue
            cls = labels[idx]
            cluster_majority[c] = Counter(cls.tolist()).most_common(1)[0][0]

        sample_pred_labels = torch.zeros_like(labels)
        for i in range(len(labels)):
            sample_pred_labels[i] = cluster_majority[sample_cluster[i].item()]

        return sample_pred_labels, K

    # =====================================================
    # Case 2: normal AP-style exemplar clustering
    # =====================================================
    # ---------- choose exemplars ----------
    exemplar_mask = exemplar_prob > exemplar_thresh
    if exemplar_mask.sum() == 0:
        # fallback: sqrt(K)
        k = max(2, int(K ** 0.5))
        exemplar_ids = torch.topk(exemplar_prob, k=k).indices
    else:
        exemplar_ids = torch.nonzero(exemplar_mask).squeeze(1)

    C = exemplar_ids.numel()
    exemplar_emb = proto_emb[exemplar_ids]   # (C, D)

    # ---------- prototype → exemplar ----------
    proto_emb_n = F.normalize(proto_emb, dim=-1)
    exemplar_emb_n = F.normalize(exemplar_emb, dim=-1)
    sim = proto_emb_n @ exemplar_emb_n.t()   # (K, C)
    proto2cluster = sim.argmax(dim=1)        # (K,)

    # ---------- global proto_id → cluster -----------
    protoid_to_cluster = {}
    for i, pid in enumerate(active_proto_ids.tolist()):
        protoid_to_cluster[pid] = proto2cluster[i].item()

    # ---------- sample-level cluster ----------
    sample_cluster = torch.zeros_like(labels)
    for i, pid in enumerate(proto_ids.tolist()):
        sample_cluster[i] = protoid_to_cluster[pid]

    # ---------- cluster → class (majority) ----------
    cluster_majority = {}
    for c in range(C):
        idx = (sample_cluster == c).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        cls = labels[idx]
        cluster_majority[c] = Counter(cls.tolist()).most_common(1)[0][0]

    # ---------- final predicted labels ----------
    sample_pred_labels = torch.zeros_like(labels)
    for i in range(len(labels)):
        sample_pred_labels[i] = cluster_majority[sample_cluster[i].item()]

    return sample_pred_labels, C


# ============================================================
# Step 3. Metrics
# ============================================================
def purity_score(y_true, y_pred):
    return (y_true == y_pred).sum() / len(y_true)


def pairwise_f_score(y_true, y_pred):
    """
    # TP: true positive pairs (both in same cluster and same class)
    # FP: false positive pairs (same cluster but different class)
    # FN: false negative pairs (same class but different cluster)
    """
    tn, fp, fn, tp = pair_confusion_matrix(y_true, y_pred).ravel()

    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return 0.0

    f_score = (2 * tp) / denominator
    return f_score


# ============================================================
# Step 4. Full inference
# ============================================================
@torch.no_grad()
def inference(ae, ap_net, loader, dataset_name, device, exemplar_thresh=0.3, min_proto_for_ap=6):
    ae.eval()
    ap_net.eval()

    stats = extract_active_protos(ae, loader, dataset_name, device)

    proto_emb = stats["proto_emb"].to(device)
    proto_usage = stats["proto_usage"].to(device)
    proto_ids = stats["proto_ids"]
    labels = stats["labels"]
    active_proto_ids = stats["active_proto_ids"]

    # ---------- AP Network ----------
    out = ap_net(proto_emb, proto_usage)
    exemplar_prob = out["exemplar_prob"]

    # ---------- clustering ----------
    pred_labels, num_clusters = cluster_by_exemplar(
        proto_emb,
        exemplar_prob,
        proto_ids,
        active_proto_ids,
        labels,
        exemplar_thresh,
        min_proto_for_ap=min_proto_for_ap
    )

    y_true = labels.numpy()
    y_pred = pred_labels.numpy()

    purity = purity_score(y_true, y_pred)
    f_score = pairwise_f_score(y_true, y_pred)

    return {
        "num_prototypes": proto_emb.size(0),
        "num_clusters": num_clusters,
        "purity": purity,
        "f_score": f_score
    }


def sensitivity_experiment(
    ae,
    ap_net,
    dataset_info,
    device,
    save_dir="./sensitivity_results"
):

    os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------------------
    # parameter range
    # --------------------------------------------------------
    thresh_list = np.arange(0.10, 0.601, 0.05)

    # --------------------------------------------------------
    # datasets
    # --------------------------------------------------------
    trained_datasets = {
        "Ls": dataset_info["Ls"],
        "Hd": dataset_info["Hd"],
        "Sf": dataset_info["Sf"]
    }

    unseen_datasets = {
        "Ly": dataset_info["Ly"],
        "Bs": dataset_info["Bs"],
        "Ce": dataset_info["Ce"]
    }

    def evaluate_group(group_dict):

        results = {}

        for short_name, info in group_dict.items():

            print(f"\nEvaluating Dataset: {short_name}")

            _, dataset = get_dataset2(
                info["data_path"],
                info["fea_path"]
            )

            loader = DataLoader(
                dataset,
                batch_size=512,
                shuffle=False
            )

            f_scores = []

            for thresh in thresh_list:
                metrics = inference(
                    ae,
                    ap_net,
                    loader,
                    info["name"],
                    device,
                    exemplar_thresh=thresh,
                    min_proto_for_ap=6
                )

                f_scores.append(metrics["f_score"])

                print(
                    f"Thresh={thresh:.2f} | "
                    f"F-score={metrics['f_score']:.4f}"
                )

            results[short_name] = f_scores

        return results

    # --------------------------------------------------------
    # run experiment
    # --------------------------------------------------------
    trained_results = evaluate_group(trained_datasets)
    unseen_results = evaluate_group(unseen_datasets)

    # ========================================================
    # plot function
    # ========================================================
    def draw_plot(results, title, save_name):

        plt.figure(figsize=(8, 6))

        for dataset_name, scores in results.items():
            plt.plot(
                thresh_list,
                scores,
                marker='o',
                linewidth=2,
                label=dataset_name
            )

        plt.xlabel("Exemplar Threshold")
        plt.ylabel("F-score")

        plt.title(title)

        plt.xticks(thresh_list)

        plt.grid(True)
        plt.legend()

        save_path = os.path.join(save_dir, save_name)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        print(f"Saved Figure: {save_path}")

        # --------------------------------------------------------
        # Figure 1: trained datasets
        # --------------------------------------------------------

    draw_plot(
        trained_results,
        title="Parameter Sensitivity on Trained Datasets",
        save_name="trained_datasets_sensitivity.png"
    )

    # --------------------------------------------------------
    # Figure 2: unseen datasets
    # --------------------------------------------------------
    draw_plot(
        unseen_results,
        title="Parameter Sensitivity on Unseen Datasets",
        save_name="unseen_datasets_sensitivity.png"
    )


def main(args):
    device = torch.device(args.device)

    # ---------- Load AE ----------
    ae = UnifiedAutoencoder(
        emb_dim=args.emb_dim,
        prototypes=args.prototypes,
        enc_layers=args.enc_layers,
        enc_heads=args.enc_heads,
        dropout=args.dropout,
        max_len=args.max_len
    )

    # --------------------------------------------------------
    # dataset information
    # --------------------------------------------------------
    dataset_info = {

        # unseen datasets
        "Ly": {
            "name": "lymphography",
            "data_path": "../datas/lymphography/test.csv",
            "fea_path": "../datas/lymphography/list.txt"
        },

        "Bs": {
            "name": "balance_scale",
            "data_path": "../datas/balance_scale/test.csv",
            "fea_path": "../datas/balance_scale/list.txt"
        },

        "Ce": {
            "name": "car_evaluation",
            "data_path": "../datas/car_evaluation/test.csv",
            "fea_path": "../datas/car_evaluation/list.txt"
        },

        # trained datasets
        "Ls": {
            "name": "lenses",
            "data_path": "../datas/lenses/test.csv",
            "fea_path": "../datas/lenses/list.txt"
        },

        "Hd": {
            "name": "heart_disease",
            "data_path": "../datas/heart_disease/test.csv",
            "fea_path": "../datas/heart_disease/list.txt"
        },

        "Sf": {
            "name": "solar_flare",
            "data_path": "../datas/solar_flare/test.csv",
            "fea_path": "../datas/solar_flare/list.txt"
        }
    }

    for _, info in dataset_info.items():

        feature_list, _ = get_dataset2(
            info["data_path"],
            info["fea_path"]
        )

        if info["name"] not in ae.encoder.adapters:
            ae.register_dataset_adapter(
                info["name"],
                input_dim=1,
                out_dim=len(feature_list)
            )

    ckpt = torch.load(args.ae_ckpt_path, map_location=device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    ae.load_state_dict(state, strict=False)
    ae.to(device)
    ae.eval()
    # ---------- AP Network ----------
    ap_net = NeuralIterativeAP(
        dim=args.emb_dim,
        iters=args.iters,
        heads=args.heads,
    ).to(device)

    ap_net.load_state_dict(torch.load(args.split_ckpt, map_location=device))
    ap_net.eval()

    # --------------------------------------------------------
    # sensitivity experiment
    # --------------------------------------------------------
    # sensitivity_experiment(
    #     ae,
    #     ap_net,
    #     dataset_info,
    #     device,
    #     save_dir="./sensitivity_results"
    # )
    # ---------- Inference ----------
    # datasets = []
    for dp, fp, name in zip(args.data_paths, args.fea_paths, args.names):
        _, dataset = get_dataset2(dp, fp)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        metrics = inference(
            ae,
            ap_net,
            loader,
            name,
            device,
            exemplar_thresh=args.exemplar_thresh,
            min_proto_for_ap=args.min_proto_for_ap
        )

        print("=" * 60)
        print(f"Dataset: {name}")
        print(f"num_prototypes: {metrics['num_prototypes']}")
        print(f"#Clusters: {metrics['num_clusters']}")
        print(f"Purity: {metrics['purity']:.4f}")
        print(f"f_score: {metrics['f_score']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae_ckpt_path", type=str, default="./Result/best_model.pt")
    parser.add_argument("--split_ckpt", type=str, default="./Result/neural_iterative_ap.pt")
    parser.add_argument("--data_paths", nargs="+", default=[
        "../datas/lymphography/test.csv",
        "../datas/balance_scale/test.csv",
        "../datas/car_evaluation/test.csv",
        "../datas/lenses/test.csv",
        "../datas/heart_disease/test.csv",
        "../datas/solar_flare/test.csv",
    ])

    parser.add_argument("--fea_paths", nargs="+", default=[
        "../datas/lymphography/list.txt",
        "../datas/balance_scale/list.txt",
        "../datas/car_evaluation/list.txt",
        "../datas/lenses/list.txt",
        "../datas/heart_disease/list.txt",
        "../datas/solar_flare/list.txt",
    ])

    parser.add_argument("--names", nargs="+", default=[
        "lymphography",
        "balance_scale",
        "car_evaluation",
        "lenses",
        "heart_disease",
        "solar_flare",
    ])

    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--prototypes", type=int, default=64)
    parser.add_argument("--enc_layers", type=int, default=2)
    parser.add_argument("--enc_heads", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)

    parser.add_argument("--exemplar_thresh", type=float, default=0.15)
    parser.add_argument("--min_proto_for_ap", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()

    main(args)