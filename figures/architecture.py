"""Hand-coded reward-model architecture diagram — AIAYN-style."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = pathlib.Path(__file__).parent / "architecture.png"

# Palette
C_INPUT = "#F3F3F6"
C_EMBED = "#FADECC"      # peach  — tokens/embeddings
C_ENCODER = "#C9E4FF"    # light blue — the backbone
C_ENCODER_B = "#3B6EE8"
C_HEAD = "#D8EEC7"       # light green — scalar head
C_REWARD = "#FFF0B2"     # yellow — scalar reward
C_LOSS = "#F9D4E3"       # pink — loss
C_EDGE = "#2A2A30"

fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 90); ax.axis("off")

def box(x, y, w, h, text, fill, edge=C_EDGE, lw=1.2, fontsize=10, fontweight="normal"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                       linewidth=lw, edgecolor=edge, facecolor=fill)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color="#111115")

def arrow(x1, y1, x2, y2, color=C_EDGE, lw=0.9, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=f"{style},head_length=2.2,head_width=1.6",
                        color=color, linewidth=lw, mutation_scale=6)
    ax.add_patch(a)

# Column x-centers: left=chosen (25), right=rejected (75)
XC, XR = 25, 75

# --- Input row: prompt + chosen / prompt + rejected ---
box(XC - 18, 78, 36, 7, "Prompt + Chosen response", C_INPUT, fontsize=10)
box(XR - 18, 78, 36, 7, "Prompt + Rejected response", C_INPUT, fontsize=10)

# --- Tokenizer + Embedding (peach) ---
box(XC - 13, 68, 26, 6, "Tokenizer + Embedding", C_EMBED, fontsize=9.5)
box(XR - 13, 68, 26, 6, "Tokenizer + Embedding", C_EMBED, fontsize=9.5)
arrow(XC, 78, XC, 74.5)
arrow(XR, 78, XR, 74.5)

# --- DistilBERT encoder (shared — shown as two instances sharing weights) ---
# Draw a dashed "shared weights" background rectangle behind both encoder blocks
shared_bg = Rectangle((8, 43), 84, 22, linewidth=1.4, edgecolor=C_ENCODER_B,
                      facecolor="#EBF3FF", linestyle=(0, (6, 3)), alpha=0.55)
ax.add_patch(shared_bg)
ax.text(50, 63.5, "Shared weights", ha="center", va="center",
        fontsize=8.5, color=C_ENCODER_B, fontweight="bold", style="italic")

# DistilBERT encoder box (one per column; shared weights indicated by the dashed background)
for col_x in (XC, XR):
    box(col_x - 13, 44, 26, 15.5, "", C_ENCODER, edge=C_ENCODER_B, lw=1.3, fontsize=0)
    ax.text(col_x, 52, "DistilBERT encoder\n6 transformer blocks\n66M params", ha="center", va="center",
            fontsize=9, color=C_ENCODER_B, fontweight="bold")
arrow(XC, 68, XC, 59.5)
arrow(XR, 68, XR, 59.5)

# --- [CLS] hidden state extraction ---
box(XC - 12, 36, 24, 5, "[CLS] hidden state (768-d)", C_INPUT, fontsize=9)
box(XR - 12, 36, 24, 5, "[CLS] hidden state (768-d)", C_INPUT, fontsize=9)
arrow(XC, 43, XC, 41)
arrow(XR, 43, XR, 41)

# --- Scalar reward head (Linear(768,1)) ---
box(XC - 12, 28, 24, 5, "Linear(768, 1)", C_HEAD, fontsize=9.5)
box(XR - 12, 28, 24, 5, "Linear(768, 1)", C_HEAD, fontsize=9.5)
arrow(XC, 36, XC, 33)
arrow(XR, 36, XR, 33)

# --- Scalar rewards ---
box(XC - 10, 19, 20, 5, r"$r_{\mathrm{chosen}} \in \mathbb{R}$", C_REWARD, fontsize=10, fontweight="bold")
box(XR - 10, 19, 20, 5, r"$r_{\mathrm{rejected}} \in \mathbb{R}$", C_REWARD, fontsize=10, fontweight="bold")
arrow(XC, 28, XC, 24)
arrow(XR, 28, XR, 24)

# --- Bradley-Terry loss ---
box(38, 7, 24, 6.5, r"$\mathcal{L} = -\log\sigma(r_{\mathrm{chosen}} - r_{\mathrm{rejected}})$",
    C_LOSS, lw=1.4, fontsize=10, fontweight="bold")
arrow(XC, 19, 45, 13.5)
arrow(XR, 19, 55, 13.5)

# --- Caption ---
ax.text(50, 1, "Two copies of a shared-weight DistilBERT encoder score the chosen and rejected responses.\nThe Bradley–Terry loss pushes $r_{\\mathrm{chosen}} > r_{\\mathrm{rejected}}$.",
        ha="center", va="center", fontsize=8.5, color="#5b5b66", style="italic")

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"OK: {OUT}")
