import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

from credit_scoring_phe.config import N_PHE
from credit_scoring_phe.evaluation.metrics import STAGE_KEYS

sns.set_style("whitegrid")


def plot_roc_curves(plain: dict, mesa: dict, fed: dict,
                    y_sub: np.ndarray) -> None:
    """ROC-кривые для трёх методов на одинаковой подвыборке."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, y_true, probs in [
        ("Plaintext", y_sub, plain["y_prob_sub"]),
        ("PHE via Mesa Agents", mesa["y_true"], mesa["y_prob"]),
        ("Federated (PHE)", y_sub, fed["y_prob_sub"]),
    ]:
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {N_PHE} samples (same subset)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_stage_latencies(stage_means: dict) -> None:
    """Столбчатая диаграмма поэтапных задержек Mesa-пайплайна."""
    stage_labels = {
        "monitoring_ms": "1. Monitoring",
        "encryption_ms": "2. Encryption",
        "transmission_ms": "3. Transmission",
        "analysis_ms": "4. Analysis",
        "decrypt_activate_ms": "5. Decrypt+Sigmoid",
    }

    labels = [stage_labels[k] for k in STAGE_KEYS]
    values = [stage_means[k] for k in STAGE_KEYS]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=sns.color_palette("tab10", len(labels)))
    ax.set_ylabel("Avg latency (ms)")
    ax.set_title("Mesa Pipeline — Per-Stage Latency Breakdown")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9
        )

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()


def plot_latency_comparison(latencies_plain: list, latencies_mesa: list) -> None:
    """Сравнение общей латентности: Plaintext vs PHE Mesa."""
    lat_names = ["Plaintext", "PHE (Mesa)"]
    lat_means = [np.mean(latencies_plain), np.mean(latencies_mesa)]
    lat_stds = [np.std(latencies_plain), np.std(latencies_mesa)]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(lat_names, lat_means, yerr=lat_stds, capsize=6)
    ax.set_ylabel("Latency per sample (ms)")
    ax.set_title("Inference Latency: Plaintext vs PHE Agent Pipeline")

    for bar, val in zip(bars, lat_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(lat_stds) * 0.05,
            f"{val:.1f} ms", ha="center", va="bottom", fontsize=11
        )

    plt.tight_layout()
    plt.show()
