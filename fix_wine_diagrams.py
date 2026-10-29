from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("wine_diagrams_fixed")
OUT.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 9


def draw_flowchart(title, nodes, edges, filename, figsize=(14, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Draw boxes
    for key, (x, y, w, h, label, color) in nodes.items():
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            linewidth=1.1,
            edgecolor="#40566f",
            facecolor=color,
            zorder=2
        ))
        ax.text(
            x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=8.5, wrap=True, zorder=3
        )

    def boundary(node, point):
        x, y, w, h, *_ = node
        cx, cy = x + w / 2, y + h / 2
        dx, dy = point[0] - cx, point[1] - cy

        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return cx, cy

        sx = (w / 2) / abs(dx) if abs(dx) > 1e-9 else float("inf")
        sy = (h / 2) / abs(dy) if abs(dy) > 1e-9 else float("inf")
        scale = min(sx, sy)
        return cx + dx * scale, cy + dy * scale

    # Connect box edges; use elbow routing for cross-row links
    for source, target in edges:
        a, b = nodes[source], nodes[target]
        ax0, ay0, aw, ah, *_ = a
        bx0, by0, bw, bh, *_ = b

        ac = (ax0 + aw / 2, ay0 + ah / 2)
        bc = (bx0 + bw / 2, by0 + bh / 2)

        if abs(ac[1] - bc[1]) < 0.35:
            start = boundary(a, bc)
            end = boundary(b, ac)
            arrow = FancyArrowPatch(
                start, end, arrowstyle="-|>",
                mutation_scale=13, linewidth=1,
                color="#526779", shrinkA=2, shrinkB=3,
                zorder=1
            )
            ax.add_patch(arrow)

        elif abs(ac[0] - bc[0]) < 0.35:
            start = boundary(a, bc)
            end = boundary(b, ac)
            arrow = FancyArrowPatch(
                start, end, arrowstyle="-|>",
                mutation_scale=13, linewidth=1,
                color="#526779", shrinkA=2, shrinkB=3,
                zorder=1
            )
            ax.add_patch(arrow)

        else:
            # Route through a clear elbow outside the boxes
            if bc[1] < ac[1]:
                start = boundary(a, (ac[0], bc[1]))
                end = boundary(b, (bc[0], ac[1]))
                mid_y = (start[1] + end[1]) / 2 - 0.12
            else:
                start = boundary(a, (bc[0], ac[1]))
                end = boundary(b, (ac[0], bc[1]))
                mid_y = (start[1] + end[1]) / 2 + 0.12

            # First segment
            ax.plot(
                [start[0], start[0]],
                [start[1], mid_y],
                color="#526779", linewidth=1, zorder=1
            )
            ax.plot(
                [start[0], end[0]],
                [mid_y, mid_y],
                color="#526779", linewidth=1, zorder=1
            )
            # Arrow on final segment
            ax.add_patch(FancyArrowPatch(
                (end[0], mid_y), end,
                arrowstyle="-|>", mutation_scale=13,
                linewidth=1, color="#526779",
                shrinkA=0, shrinkB=3, zorder=1
            ))

    ax.set_title(title, fontsize=13, fontweight="bold", pad=18)
    fig.tight_layout(pad=1.5)
    fig.savefig(OUT / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Created:", OUT / filename)


# FIGURE 1: RESEARCH WORKFLOW
nodes1 = {
    "data": (0.4, 5.5, 2.3, 1.0,
             "UCI Wine Quality Dataset\nRed / White Wine", "#E2EDFA"),
    "inspect": (3.4, 5.5, 2.3, 1.0,
                "Data Inspection\nMissing Values / Classes", "#E2EDFA"),
    "split": (6.4, 5.5, 2.3, 1.0,
              "Stratified Train-Test Split", "#FFF0D5"),
    "prep": (9.4, 5.5, 2.8, 1.0,
             "Preprocessing\nScaling / Optional SMOTE", "#FFF0D5"),
    "models": (1.8, 3.0, 2.5, 1.0,
               "Train Classifiers\nDT / RF / SVM / KNN", "#DCEFD9"),
    "tune": (5.0, 3.0, 2.5, 1.0,
             "Cross-Validation\nHyperparameter Tuning", "#DCEFD9"),
    "eval": (8.2, 3.0, 2.5, 1.0,
             "Evaluation\nMacro-F1 / Balanced Accuracy", "#F7DFE3"),
    "result": (5.0, 0.7, 3.0, 1.0,
               "Model Comparison and Discussion", "#DCE7F8"),
}

edges1 = [
    ("data", "inspect"),
    ("inspect", "split"),
    ("split", "prep"),
    ("prep", "models"),
    ("models", "tune"),
    ("tune", "eval"),
    ("eval", "result"),
]

draw_flowchart(
    "Research Workflow for Wine Quality Prediction",
    nodes1, edges1, "Fig1_research_workflow.png"
)


# FIGURE 2: SYSTEM ARCHITECTURE
nodes2 = {
    "input": (0.4, 5.5, 2.3, 1.0,
              "Input Features\n11 Physicochemical Variables", "#E2EDFA"),
    "pre": (3.4, 5.5, 2.3, 1.0,
            "Data Preprocessing\nTrain-Only Transformations", "#FFF0D5"),
    "feature": (6.4, 5.5, 2.3, 1.0,
                "Feature Selection\nOptional Experiment", "#E9E0F6"),
    "classifier": (9.4, 5.5, 2.8, 1.0,
                   "Classifier\nDT / RF / SVM / KNN", "#DCEFD9"),
    "prediction": (9.4, 3.0, 2.8, 1.0,
                   "Predicted Quality Class", "#DCE7F8"),
    "evaluation": (5.4, 3.0, 2.8, 1.0,
                   "Metrics and Confusion Matrix", "#F7DFE3"),
    "report": (1.4, 3.0, 2.8, 1.0,
               "Comparison and Interpretation", "#E2EDFA"),
}

edges2 = [
    ("input", "pre"),
    ("pre", "feature"),
    ("feature", "classifier"),
    ("classifier", "prediction"),
    ("prediction", "evaluation"),
    ("evaluation", "report"),
]

draw_flowchart(
    "Architecture of the Wine Quality Prediction System",
    nodes2, edges2, "Fig2_system_architecture.png"
)

print("\nDone. Output folder:", OUT.resolve())