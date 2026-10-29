from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
OUT = Path("wine_outputs")
OUT.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 9
plt.rcParams["figure.facecolor"] = "white"

# ==========================================
# DATA: EXACTLY 13 ROWS PROVIDED
# ==========================================
columns = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality"
]

data = [
    [7.4, 0.7,   0,    1.9, 0.076, 11,  34,  0.9978, 3.51, 0.56,  9.4,  5],
    [7.8, 0.88,  0,    2.6, 0.098, 25,  67,  0.9968, 3.20, 0.68,  9.8,  5],
    [7.8, 0.76,  0.04, 2.3, 0.092, 15,  54,  0.9970, 3.26, 0.65,  9.8,  5],
    [11.2, 0.28, 0.56, 1.9, 0.075, 17,  60,  0.9980, 3.16, 0.58,  9.8,  6],
    [7.4, 0.7,   0,    1.9, 0.076, 11,  34,  0.9978, 3.51, 0.56,  9.4,  5],
    [7.4, 0.66,  0,    1.8, 0.075, 13,  40,  0.9978, 3.51, 0.56,  9.4,  5],
    [7.9, 0.6,   0.06, 1.6, 0.069, 15,  59,  0.9964, 3.30, 0.46,  9.4,  5],
    [7.3, 0.65,  0,    1.2, 0.065, 15,  21,  0.9946, 3.39, 0.47, 10.0,  7],
    [7.8, 0.58,  0.02, 2.0, 0.073,  9,  18,  0.9968, 3.36, 0.57,  9.5,  7],
    [7.5, 0.5,   0.36, 6.1, 0.071, 17, 102,  0.9978, 3.35, 0.80, 10.5,  5],
    [6.7, 0.58,  0.08, 1.8, 0.097, 15,  65,  0.9959, 3.28, 0.54,  9.2,  5],
    [7.5, 0.5,   0.36, 6.1, 0.071, 17, 102,  0.9978, 3.35, 0.80, 10.5,  5],
    [5.6, 0.615, 0,    1.6, 0.089, 16,  59,  0.9943, 3.58, 0.52,  9.9,  5],
]

df = pd.DataFrame(data, columns=columns)

print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

# ==========================================
# TABLE HELPER
# ==========================================
def save_table(title, table_df, filename, font_size=8):
    table_df.to_csv(
        OUT / filename.replace(".png", ".csv"),
        index=False,
        encoding="utf-8-sig"
    )

    fig_height = max(2.5, 0.42 * (len(table_df) + 1) + 1.0)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#AAB3BD")
        cell.set_linewidth(0.5)

        if row == 0:
            cell.set_facecolor("#DCE7F8")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor(
                "#FFFFFF" if row % 2 else "#F3F6F9"
            )

    ax.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        pad=15
    )

    fig.tight_layout()
    fig.savefig(
        OUT / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("Created:", OUT / filename)
    print("Created:", OUT / filename.replace(".png", ".csv"))


# ==========================================
# TABLE 3: DESCRIPTIVE STATISTICS
# ==========================================
features = [col for col in df.columns if col != "quality"]

stats = df[features].describe().T.reset_index()
stats.columns = [
    "Feature", "Count", "Mean", "Std",
    "Min", "Q1", "Median", "Q3", "Max"
]

stats["Count"] = stats["Count"].astype(int)

numeric_cols = [
    "Mean", "Std", "Min", "Q1",
    "Median", "Q3", "Max"
]
stats[numeric_cols] = stats[numeric_cols].round(4)

save_table(
    "Table 3. Descriptive Statistics of Wine Quality Features",
    stats,
    "Table3_descriptive_statistics.png",
    font_size=7.5
)

# ==========================================
# TABLE 4: QUALITY CLASS DISTRIBUTION
# ==========================================
quality_counts = (
    df["quality"]
    .value_counts()
    .sort_index()
    .rename_axis("Quality Score")
    .reset_index(name="Number of Samples")
)

quality_counts["Percentage (%)"] = (
    quality_counts["Number of Samples"]
    / len(df) * 100
).round(2)

# Add total row
total_row = pd.DataFrame([{
    "Quality Score": "Total",
    "Number of Samples": len(df),
    "Percentage (%)": 100.00
}])

quality_counts["Quality Score"] = (
    quality_counts["Quality Score"].astype(str)
)

quality_counts = pd.concat(
    [quality_counts, total_row],
    ignore_index=True
)

save_table(
    "Table 4. Wine Quality Class Distribution",
    quality_counts,
    "Table4_quality_distribution.png",
    font_size=8
)

# ==========================================
# FINAL SUMMARY
# ==========================================
print("\nAll tables were generated.")
print("Output folder:", OUT.resolve())
print("\nQuality distribution:")
print(quality_counts.to_string(index=False))