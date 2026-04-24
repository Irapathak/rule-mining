import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
SCALABILITY_ALGORITHMS = ["RW", "FPGrowth"]
REPEATS_PER_CONFIG = 3
ROW_PERCENTAGES = [x / 10 for x in range(1, 11)]
DATASETS = ["acs", "so"]

# Wong's color-blind safe palette (matches generate_figures)
WONG = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "yellow": "#F0E442",
    "skyblue": "#56B4E9",
    "red": "#D55E00",
    "pink": "#CC79A7",
    "black": "#000000",
}


def generate_graphs():
    results_dir = Path(__file__).resolve().parent.parent / "graphs"

    for ds in DATASETS:
        row_file = results_dir / f"{ds}_scalability_rows.csv"
        if not row_file.exists():
            print(f"⚠ Skipping {ds}: {row_file} not found")
            continue

        df_rows = pd.read_csv(row_file)

        # Rename: FPGrowth -> BruteForce, RW_unlearning -> RW
        df_rows["algorithm"] = df_rows["algorithm"].replace({
            "FPGrowth": "BruteForce",
            "RW_unlearning": "RW",
        })

        df_rows_agg = (
            df_rows.groupby(["algorithm", "dataset_percentage"])["run_time_seconds"]
            .mean()
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.tick_params(axis="both", labelsize=11)
        ax.xaxis.label.set_size(13)
        ax.xaxis.label.set_weight("bold")
        ax.yaxis.label.set_size(13)
        ax.yaxis.label.set_weight("bold")

        for i, algo in enumerate(df_rows_agg["algorithm"].unique()):
            sub = df_rows_agg[df_rows_agg["algorithm"] == algo].sort_values("dataset_percentage")
            color = list(WONG.values())[i % len(WONG)]
            marker = "s" if algo == "BruteForce" else "o"
            ax.plot(
                sub["dataset_percentage"],
                sub["run_time_seconds"],
                marker=marker,
                linewidth=2.5,
                color=color,
                label=algo,
                markersize=8,
            )

        ax.set_xlabel("Dataset Percentage")
        ax.set_ylabel("Avg Runtime (s)")
        ax.set_xticks(ROW_PERCENTAGES)
        ax.set_xticklabels([f"{int(x * 100)}%" for x in ROW_PERCENTAGES])
        ax.legend(loc="best", fontsize=11, frameon=True, title=None)
        ax.grid(True, alpha=0.3)

        if ds == "so":
            ax.set_yscale("log")

        out_path = results_dir / f"{ds}_scalability_rows_graph.pdf"
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    generate_graphs()
