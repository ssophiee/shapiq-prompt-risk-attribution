import matplotlib.pyplot as plt
import numpy as np


def plot_results(cot_steps: list[str], sv, ksii, title: str = "CoT Step Attribution") -> None:
    n = len(cot_steps)
    sv_vals = sv.values[1 : n + 1][::-1]
    step_labels = [f"Step {i + 1}" for i in range(n)][::-1]

    imat = np.zeros((n, n))
    for interaction, value in zip(ksii.interactions, ksii.values):
        if len(interaction) == 2:
            i, j = interaction
            imat[i, j] = value
            imat[j, i] = value

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, n)))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(title, color="white", fontweight="bold", fontsize=13, y=1.01)

    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in sv_vals]
    bars = ax1.barh(step_labels, sv_vals, color=colors, edgecolor="white", height=0.5)
    ax1.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.set_xlabel("Shapley Value", color="white")
    ax1.set_title("First-order SVs\n(can be misleading — see heatmap)", color="white")
    ax1.set_facecolor("#1a1a2e")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_visible(False)
    for bar, val in zip(bars, sv_vals):
        ax1.text(
            val + (0.002 if val >= 0 else -0.002),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            color="white",
            fontsize=8,
        )

    step_labels_fwd = [f"Step {i + 1}" for i in range(n)]
    vmax = max(abs(imat).max(), 0.001)
    im = ax2.imshow(imat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(step_labels_fwd, color="white", rotation=30, ha="right")
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(step_labels_fwd, color="white")
    ax2.set_title("k-SII Pairwise Interactions\n(+ = synergy, - = redundancy)", color="white")
    ax2.set_facecolor("#1a1a2e")
    for i in range(n):
        for j in range(n):
            if i != j:
                ax2.text(j, i, f"{imat[i, j]:+.2f}", ha="center", va="center", color="black", fontsize=8)
    plt.colorbar(im, ax=ax2, label="Interaction value")
    plt.tight_layout()
    fname = title.replace(" ", "_") + ".png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"Saved: {fname}")
