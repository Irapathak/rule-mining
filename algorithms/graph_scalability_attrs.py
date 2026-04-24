import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
SCALABILITY_ALGORITHMS = ["RW", "FPGrowth"]
REPEATS_PER_CONFIG = 3
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
        attr_file = results_dir / f"{ds}_scalability_attributes.csv"
        if not attr_file.exists():
            print(f"⚠ Skipping {ds}: {attr_file} not found")
            continue

        df_attrs = pd.read_csv(attr_file)

        # Rename: FPGrowth -> BruteForce, RW_unlearning -> RW
        df_attrs["algorithm"] = df_attrs["algorithm"].replace({
            "FPGrowth": "BruteForce",
            "RW_unlearning": "RW",
        })

        # Filter to num_attributes >= 0 and convert to actual count (starts at 3)
        BASE_ATTRS = 3
        df_attrs = df_attrs[df_attrs["num_attributes"] >= 0].copy()
        df_attrs["actual_num_attrs"] = df_attrs["num_attributes"] + BASE_ATTRS

        df_attrs_agg = (
            df_attrs.groupby(["algorithm", "actual_num_attrs"])["run_time_seconds"]
            .mean()
            .reset_index()
        )
        x_vals = sorted(df_attrs_agg["actual_num_attrs"].unique())

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.tick_params(axis="both", labelsize=11)
        ax.xaxis.label.set_size(13)
        ax.xaxis.label.set_weight("bold")
        ax.yaxis.label.set_size(13)
        ax.yaxis.label.set_weight("bold")

        for i, algo in enumerate(df_attrs_agg["algorithm"].unique()):
            sub = df_attrs_agg[df_attrs_agg["algorithm"] == algo].sort_values("actual_num_attrs")
            color = list(WONG.values())[i % len(WONG)]
            marker = "s" if algo == "BruteForce" else "o"
            ax.plot(
                sub["actual_num_attrs"],
                sub["run_time_seconds"],
                marker=marker,
                linewidth=2.5,
                color=color,
                label=algo,
                markersize=8,
            )

        ax.set_xlabel("Num of Attributes")
        ax.set_ylabel("Avg Runtime (s)")
        ax.set_xlim(left=BASE_ATTRS)
        ax.set_xticks(x_vals)
        ax.legend(loc="best", fontsize=11, frameon=True, title=None)
        ax.grid(True, alpha=0.3)

        out_path = results_dir / f"{ds}_scalability_attributes_graph.pdf"
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    generate_graphs()
