import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Confusion matrix (TEST) - raw counts
# --------------------------------------------------
cm = np.array([
    [30667, 1415,  132,   5,   0],
    [ 2167, 5418,  400,  11,   0],
    [  179,  720, 1718,  43,   0],
    [    3,   45,  146, 503,  43],
    [    0,    0,    0,  69, 129],
], dtype=int)

# Class names (order must match cm rows/cols)
class_names = [
    "Normal",
    "Mild Disturbance",
    "Moderate Storm",
    "Severe Storm",
    "Extreme Event"
]

# --------------------------------------------------
# Summary metrics (already computed elsewhere)
# --------------------------------------------------
metrics = {
    "Accuracy": 0.88,
    "Macro-F1": 0.74914806692127,
    "Weighted-F1": 0.88,
    "Macro Recall": (0.95 + 0.68 + 0.65 + 0.68 + 0.65) / 5.0,
    "Macro Precision": (0.93 + 0.71 + 0.72 + 0.80 + 0.75) / 5.0,
}

# Small tags for file naming / titles
MODEL_TAG  = "cnn"
TITLE_LINE = "CNN"

# --------------------------------------------------
# Normalize confusion matrix by row (true-class normalization)
# Each row sums to 1.0 -> easier to compare across classes
# --------------------------------------------------
row_sums = cm.sum(axis=1, keepdims=True)
cm_norm = cm / np.maximum(row_sums, 1)

# --------------------------------------------------
# Plot normalized confusion matrix (heatmap)
# --------------------------------------------------
plt.figure(figsize=(7.2, 6.0))
im = plt.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)

plt.title(f"Normalized Confusion Matrix\n{TITLE_LINE}")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.xticks(range(len(class_names)), class_names, rotation=30, ha="right")
plt.yticks(range(len(class_names)), class_names)

# Write normalized values into each cell
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        plt.text(
            j, i,
            f"{cm_norm[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=10
        )

plt.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(f"{MODEL_TAG}_test_confusion_matrix_normalized.png", dpi=200)
plt.show()

# --------------------------------------------------
# Build a simple metrics table figure (for slides)
# --------------------------------------------------
metrics_df = pd.DataFrame(
    [{"Metric": k, "Value": v} for k, v in metrics.items()]
).sort_values("Metric").reset_index(drop=True)

# Format numeric values nicely
metrics_df["Value"] = metrics_df["Value"].map(lambda x: f"{x:.3f}")

fig, ax = plt.subplots(figsize=(6.5, 2.2))
ax.axis("off")

table = ax.table(
    cellText=metrics_df.values,
    colLabels=metrics_df.columns,
    cellLoc="left",
    colLoc="left",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.35)

ax.set_title("Evaluation Metrics", pad=12)
plt.tight_layout()
plt.savefig(f"{MODEL_TAG}_test_metrics_table.png", dpi=200)
plt.show()

# --------------------------------------------------
# Compute per-class precision / recall / f1 from confusion matrix
# --------------------------------------------------
tp = np.diag(cm)
fp = cm.sum(axis=0) - tp
fn = cm.sum(axis=1) - tp

precision = tp / np.maximum(tp + fp, 1)
recall    = tp / np.maximum(tp + fn, 1)
f1        = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
support   = cm.sum(axis=1)

# Put per-class metrics into a DataFrame
df = pd.DataFrame({
    "Class": class_names,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
    "Support": support
})

# Add macro row at the bottom (simple mean over classes)
macro = pd.DataFrame([{
    "Class": "Macro",
    "Precision": df["Precision"].mean(),
    "Recall": df["Recall"].mean(),
    "F1-score": df["F1-score"].mean(),
    "Support": int(df["Support"].sum())
}])

df2 = pd.concat([df, macro], ignore_index=True)

# --------------------------------------------------
# Prepare a formatted version for the table
# --------------------------------------------------
df_fmt = df2.copy()
for c in ["Precision", "Recall", "F1-score"]:
    df_fmt[c] = df_fmt[c].map(lambda x: f"{x:.2f}")
df_fmt["Support"] = df_fmt["Support"].map(
    lambda x: f"{int(x):,}" if isinstance(x, (int, np.integer)) else f"{x:,}"
)

# --------------------------------------------------
# Helper functions for a soft color gradient (based on F1-score)
# --------------------------------------------------
def lerp(a, b, t):
    return a + (b - a) * t

def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=float)

def rgb_to_hex(rgb):
    rgb = np.clip(rgb, 0, 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*rgb)

# Light -> mid -> green-ish palette
C_LOW  = hex_to_rgb("#FCE8D5")
C_MID  = hex_to_rgb("#FFF7CC")
C_HIGH = hex_to_rgb("#DDF3E3")

def f1_color(v, vmin=None, vmax=None):
    # Map v into [0,1] and pick a smooth gradient color
    if vmin is None: vmin = float(np.min(f1))
    if vmax is None: vmax = float(np.max(f1))
    t = 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)
    if t < 0.5:
        rgb = lerp(C_LOW, C_MID, t / 0.5)
    else:
        rgb = lerp(C_MID, C_HIGH, (t - 0.5) / 0.5)
    return rgb_to_hex(rgb)

# --------------------------------------------------
# Create a nice-looking per-class table figure
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(10.2, 3.2))
ax.axis("off")

col_labels = list(df_fmt.columns)
cell_text  = df_fmt.values.tolist()

table = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
    colLoc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(11.5)
table.scale(1, 1.65)

# Column widths (helps the table look balanced on slides)
col_widths = [0.30, 0.16, 0.16, 0.16, 0.20]
for c, w in enumerate(col_widths):
    for r in range(len(df_fmt) + 1):
        table[(r, c)].set_width(w)

# Light borders + header styling
EDGE = "#D9D9D9"
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor(EDGE)
    cell.set_linewidth(0.6)

    if r == 0:
        cell.set_facecolor("#F2F2F2")
        cell.set_text_props(weight="bold", color="black")
        cell.set_linewidth(0.9)
    else:
        cell.set_facecolor("#FFFFFF" if r % 2 == 1 else "#FAFAFA")

# Right-align support column (numbers read nicer)
support_col = col_labels.index("Support")
for r in range(1, len(df_fmt) + 1):
    table[(r, support_col)]._loc = "right"
    table[(r, support_col)].PAD = 0.02

# Colorize the F1-score column, highlight best class, and style macro row
f1_col = col_labels.index("F1-score")
best_idx = int(np.argmax(f1))

for i in range(len(df_fmt)):
    r = i + 1
    if df2.loc[i, "Class"] == "Macro":
        table[(r, f1_col)].set_facecolor("#E8EEF9")
        table[(r, 0)].set_text_props(weight="bold")
        table[(r, 0)].set_facecolor("#E8EEF9")
        for cc in range(1, len(col_labels)):
            table[(r, cc)].set_text_props(weight="bold")
        continue

    v = float(df2.loc[i, "F1-score"])
    table[(r, f1_col)].set_facecolor(f1_color(v))

    # Bold the best-performing class row (based on F1)
    if i == best_idx:
        for cc in range(len(col_labels)):
            table[(r, cc)].set_text_props(weight="bold")

ax.set_title(f"Per-Class Metrics ({TITLE_LINE})", fontsize=13.5, pad=14)

plt.tight_layout()
plt.savefig(
    f"{MODEL_TAG}_test_per_class_metrics_ultra.png",
    dpi=350,
    bbox_inches="tight",
    facecolor="white"
)
plt.show()
