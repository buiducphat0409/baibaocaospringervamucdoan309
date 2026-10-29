from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd

# =========================================================
# CONFIGURATION
# =========================================================
OUT = Path("wine_outputs")
OUT.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 9
plt.rcParams["figure.facecolor"] = "white"


# =========================================================
# DIAGRAM HELPER
# =========================================================
def draw_flowchart(title, nodes, edges, filename,
                   figsize=(12, 6)):
    """
    nodes: {id: (x, y, width, height, label, color)}
    Coordinates use a 0..12 x 0..7 canvas.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    for node_id, (x, y, w, h, label, color) in nodes.items():
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.10",
            linewidth=1.1,
            edgecolor="#40566f",
            facecolor=color
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=8.5, wrap=True
        )

    for source, target in edges:
        x, y, w, h, *_ = nodes[source]
        tx, ty, tw, th, *_ = nodes[target]

        start = (x + w / 2, y + h / 2)
        end = (tx + tw / 2, ty + th / 2)

        arrow = FancyArrowPatch(
            start, end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1,
            color="#526779",
            shrinkA=29,
            shrinkB=29,
            connectionstyle="arc3,rad=0.0"
        )
        ax.add_patch(arrow)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# FIGURE 1: RESEARCH WORKFLOW
# =========================================================
nodes1 = {
    "data": (0.4, 4.5, 2.0, 1.0,
             "UCI Wine Quality Dataset\nRed / White Wine", "#E2EDFA"),
    "inspect": (3.1, 4.5, 2.0, 1.0,
                "Data Inspection\nMissing Values / Classes", "#E2EDFA"),
    "split": (5.8, 4.5, 2.0, 1.0,
              "Stratified Train-Test Split", "#FFF0D5"),
    "prep": (8.5, 4.5, 2.6, 1.0,
             "Preprocessing\nScaling / Optional SMOTE", "#FFF0D5"),
    "models": (3.0, 2.2, 2.2, 1.0,
               "Train Classifiers\nDT / RF / SVM / KNN", "#DCEFD9"),
    "tune": (6.0, 2.2, 2.2, 1.0,
             "Cross-Validation\nHyperparameter Tuning", "#DCEFD9"),
    "eval": (9.0, 2.2, 2.2, 1.0,
             "Evaluation\nMacro-F1 / Balanced Accuracy", "#F7DFE3"),
    "result": (5.0, 0.35, 3.0, 0.85,
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


# =========================================================
# FIGURE 2: SYSTEM ARCHITECTURE
# =========================================================
nodes2 = {
    "input": (0.3, 4.6, 2.0, 1.0,
              "Input Features\n11 Physicochemical Variables", "#E2EDFA"),
    "pre": (3.0, 4.6, 2.0, 1.0,
            "Data Preprocessing\nTrain-Only Transformations", "#FFF0D5"),
    "feature": (5.7, 4.6, 2.0, 1.0,
                "Feature Selection\nOptional Experiment", "#E9E0F6"),
    "classifier": (8.4, 4.6, 2.8, 1.0,
                   "Classifier\nDT / RF / SVM / KNN", "#DCEFD9"),
    "prediction": (8.4, 2.3, 2.8, 1.0,
                   "Predicted Quality Class", "#DCE7F8"),
    "evaluation": (4.7, 2.3, 2.5, 1.0,
                   "Metrics and Confusion Matrix", "#F7DFE3"),
    "report": (1.1, 2.3, 2.5, 1.0,
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


# =========================================================
# TABLE HELPER
# =========================================================
def save_table(title, columns, rows, filename,
               widths=None, font_size=8):
    df = pd.DataFrame(rows, columns=columns)

    # CSV can be opened and edited in Excel.
    df.to_csv(OUT / filename.replace(".png", ".csv"),
              index=False, encoding="utf-8-sig")

    fig_height = max(2.0, 0.48 * len(df) + 1.25)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=widths
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.45)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#AAB3BD")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#DCE7F8")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor(
                "#FFFFFF" if r % 2 else "#F3F6F9"
            )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Created:", OUT / filename)


# =========================================================
# TABLE 1: COMMERCIAL PRODUCTS
# Values are examples from the paper draft.
# Verify current prices before publication.
# =========================================================
save_table(
    "Table 1. Examples of Commercial Wine Products",
    ["Product", "Origin", "Grape / Style",
     "Alcohol", "Approx. Price (VND)"],
    [
        ["Mapu Cabernet Sauvignon", "Chile",
         "Cabernet Sauvignon", "Verify", "320,000"],
        ["Montes Classic Series Cabernet Sauvignon", "Chile",
         "Cabernet Sauvignon", "14%", "Verify"],
        ["Marques de Casa Concha Cabernet Sauvignon", "Chile",
         "Cabernet Sauvignon", "14.5%", "795,000"],
        ["Escudo Rojo Cabernet Sauvignon", "Chile",
         "Cabernet Sauvignon", "13.5%", "Verify"],
        ["Mouton Cadet Bordeaux Reserve Médoc", "France",
         "Cabernet Sauvignon / Merlot / Cabernet Franc",
         "Verify", "530,000"],
    ],
    "Table1_commercial_products.png",
    font_size=8
)


# =========================================================
# TABLE 2: SOURCE CODE
# Do not invent repository URLs or claim they were tested.
# =========================================================
save_table(
    "Table 2. Source Code and Implementation Resources",
    ["Resource", "Purpose", "Main Use", "Verification"],
    [
        ["tqi2/Wine-Quality-Prediction",
         "Classical classification",
         "DT, RF, SVC, KNN",
         "Verify repository URL and current contents"],
        ["Scikit-learn documentation",
         "Model implementation",
         "Pipelines, preprocessing, evaluation",
         "Official documentation"],
        ["Imbalanced-learn documentation",
         "Class imbalance",
         "SMOTE inside training pipelines",
         "Official documentation"],
        ["UCI Wine Quality repository",
         "Dataset access",
         "Red and white wine data",
         "Dataset ID 186"],
    ],
    "Table2_source_resources.png",
    font_size=8
)


# =========================================================
# TABLE 3: RESEARCH APPROACHES
# =========================================================
save_table(
    "Table 3. Summary of Research Approaches",
    ["Study", "Research Focus", "Methods / Contribution"],
    [
        ["Cortez et al. (2009)",
         "Wine quality prediction",
         "Relates physicochemical properties to quality"],
        ["Gupta (2018)",
         "Feature importance",
         "Examines relevant predictors"],
        ["Breiman (2001)",
         "Ensemble learning",
         "Introduces Random Forest"],
        ["Chawla et al. (2002)",
         "Class imbalance",
         "Introduces SMOTE"],
        ["Jain et al. (2023)",
         "Wine classification",
         "Model comparison and feature selection"],
        ["Sinaga et al. (2025)",
         "White wine classification",
         "SMOTE, feature selection, parameter tuning"],
        ["Khan et al. (2026)",
         "Advanced prediction",
         "Ensembles and explainability, as described in draft"],
    ],
    "Table3_research_summary.png",
    font_size=7.5
)


# =========================================================
# TABLE 4: DATASET FEATURES
# =========================================================
save_table(
    "Table 4. UCI Wine Quality Dataset Characteristics",
    ["Item", "Description"],
    [
        ["Dataset source", "UCI Machine Learning Repository, ID 186"],
        ["Red wine observations", "1,599"],
        ["White wine observations", "4,898"],
        ["Combined observations", "6,497"],
        ["Number of input features", "11"],
        ["Target variable", "quality"],
        ["Input type", "Physicochemical numerical measurements"],
        ["Target use", "Regression or classification"],
        ["Important limitation",
         "No brand, market price, or grape-variety fields"],
    ],
    "Table4_dataset_characteristics.png",
    widths=[0.30, 0.70],
    font_size=8
)


# =========================================================
# TABLE 5: ALGORITHM COMPARISON
# =========================================================
save_table(
    "Table 5. Main Classification Algorithms",
    ["Algorithm", "Basic Principle", "Scaling Sensitivity",
     "Strength", "Limitation"],
    [
        ["Decision Tree",
         "Recursive feature-based splits",
         "Low",
         "Interpretable rules",
         "May overfit"],
        ["Random Forest",
         "Ensemble of decision trees",
         "Low",
         "Robust baseline and feature importance",
         "Less transparent than one tree"],
        ["SVM",
         "Maximum-margin classification",
         "High",
         "Effective with suitable feature scaling",
         "Parameter and kernel sensitivity"],
        ["KNN",
         "Nearest-neighbor voting",
         "High",
         "Simple, distance-based baseline",
         "Sensitive to scale and dimensionality"],
        ["XGBoost",
         "Gradient-boosted trees",
         "Generally low",
         "Advanced ensemble baseline",
         "Requires careful tuning"],
    ],
    "Table5_algorithm_comparison.png",
    font_size=7.5
)


# =========================================================
# TABLE 6: EVALUATION METRICS
# =========================================================
save_table(
    "Table 6. Evaluation Metrics for Multiclass Classification",
    ["Metric", "What It Measures", "Role in This Study"],
    [
        ["Accuracy",
         "Overall proportion of correct predictions",
         "General performance"],
        ["Balanced Accuracy",
         "Average recall across classes",
         "Useful for class imbalance"],
        ["Macro-Precision",
         "Unweighted mean of class precision",
         "Equal class-level weighting"],
        ["Macro-Recall",
         "Unweighted mean of class recall",
         "Minority-class sensitivity"],
        ["Macro-F1",
         "Unweighted mean of class F1",
         "Primary comparison metric"],
        ["Weighted-F1",
         "F1 averaged by class support",
         "Performance weighted by frequency"],
        ["ROC-AUC",
         "Ranking discrimination from scores",
         "Use a defined multiclass strategy"],
        ["Confusion Matrix",
         "Counts of true versus predicted classes",
         "Identify specific class errors"],
        ["Log Loss",
         "Quality of predicted probabilities",
         "Evaluate probabilistic predictions"],
    ],
    "Table6_evaluation_metrics.png",
    font_size=8
)


print("\nDone. All figures and tables are in:", OUT.resolve())