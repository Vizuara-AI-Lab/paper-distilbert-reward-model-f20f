"""Hand-coded RLHF pipeline diagram — AIAYN-style pastel boxes + clean arrows + exact text."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = pathlib.Path(__file__).parent / "rlhf-pipeline.png"

# AIAYN-style pastel palette
C_STAGE1 = "#FADECC"  # peach
C_STAGE2 = "#C9E4FF"  # light blue (highlighted — our focus)
C_STAGE2_B = "#3B6EE8"  # stronger blue border for the highlighted stage
C_STAGE3 = "#D8EEC7"  # light green
C_IO = "#F3F3F6"      # light gray for inputs/outputs
C_EDGE = "#2A2A30"

fig, ax = plt.subplots(figsize=(11, 4.2), dpi=300)
ax.set_xlim(0, 110); ax.set_ylim(0, 42); ax.axis("off")

def box(x, y, w, h, text, fill, edge=C_EDGE, lw=1.2, fontsize=10, fontweight="normal"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2",
                       linewidth=lw, edgecolor=edge, facecolor=fill)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color="#111115")

def arrow(x1, y1, x2, y2, color=C_EDGE, lw=0.9):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>,head_length=2.2,head_width=1.6",
                        color=color, linewidth=lw,
                        connectionstyle="arc3,rad=0", mutation_scale=6)
    ax.add_patch(a)

def stage_label(x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=10, fontweight="bold", color="#1a1a22")

# --- Stage 1: SFT ---
stage_label(15, 38, "Stage 1 — Supervised Fine-Tuning")
box( 3, 27, 24, 6, "SFT", C_STAGE1, fontsize=13, fontweight="bold")
box( 3, 19, 11, 5, "Pretrained\nLM", C_IO, fontsize=9)
box(16, 19, 11, 5, "Demonstrations", C_IO, fontsize=9)
box( 8, 10, 14, 5, "SFT Policy", C_STAGE1, fontsize=10, fontweight="bold")
arrow( 9, 24, 11, 27)
arrow(21, 24, 19, 27)
arrow(15, 27, 15, 15)

# --- Stage 2: Reward Modeling (HIGHLIGHTED) ---
stage_label(55, 38, "Stage 2 — Reward Modeling (this paper)")
# Background highlight rect
hl = Rectangle((31, 8.5), 48, 28, linewidth=1.8, edgecolor=C_STAGE2_B,
               facecolor="#EBF3FF", linestyle="--", alpha=0.55)
ax.add_patch(hl)
box(37, 27, 20, 6, "DistilBERT\nencoder + head", C_STAGE2, edge=C_STAGE2_B, lw=1.6,
    fontsize=10, fontweight="bold")
box(34, 19, 10, 5, "SFT Policy", C_IO, fontsize=9)
box(47, 19, 12, 5, "Preference\npairs", C_IO, fontsize=9)
box(42, 10, 14, 5, "Reward r(x)", C_STAGE2, edge=C_STAGE2_B, lw=1.6, fontsize=10, fontweight="bold")
arrow(39, 24, 43, 27)
arrow(53, 24, 49, 27)
arrow(47, 27, 49, 15)
# Bradley-Terry loss pill
box(61, 10, 16, 5, "Bradley–Terry\nloss", C_STAGE2, edge=C_STAGE2_B, lw=1.4, fontsize=9)
arrow(56, 12.5, 61, 12.5)

# --- Stage 3: PPO ---
stage_label(96, 38, "Stage 3 — PPO policy")
box(84, 27, 22, 6, "PPO", C_STAGE3, fontsize=13, fontweight="bold")
box(82, 19, 10, 5, "SFT Policy", C_IO, fontsize=9)
box(94, 19, 12, 5, "Reward model", C_IO, fontsize=9)
box(88, 10, 14, 5, "Aligned policy", C_STAGE3, fontsize=10, fontweight="bold")
arrow(87, 24, 91, 27)
arrow(100, 24, 97, 27)
arrow(95, 27, 95, 15)

# --- Inter-stage arrows: Stage 1 → Stage 2 (via SFT Policy) ---
arrow(22, 12.5, 34, 21)  # SFT Policy to Stage 2's SFT Policy input
# Stage 2 → Stage 3 (Reward model)
arrow(77, 12.5, 94, 21)  # Reward out to Stage 3 reward input
# Stage 1 → Stage 3 (SFT Policy reuse)
arrow(15, 10, 82, 21, color="#888890", lw=0.9)

# --- Footer caption strip ---
ax.text(55, 2, "This paper studies Stage 2 (Reward Modeling). Stages 1 and 3 are shown for context.",
        ha="center", va="center", fontsize=9, color="#5b5b66", style="italic")

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"OK: {OUT}")
