"""
Generate CodonBench V25 Figure 1 (conceptual framework figure).

4 panels:
  a: Two information channels in CDS (dual-channel + information decomposition)
  b: Tokenization as inductive bias (codon / BPE / char three rows)
  c: Nonlinear depth ladder (core finding, visual center, full-width)
  d: CodonBench framework flow (placeholder — needs hand-drawing)

Based on: paper/FIG1-DESIGN.md
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

TEAL = "#2B5F8A"
ORANGE = "#E8833A"
SLATE = "#6B7B8D"
LAVENDER = "#9B8EC4"
LIGHT_TEAL = "#D4E6F1"
LIGHT_ORANGE = "#FDEBD0"
WHITE = "#FFFFFF"
BG = "#FAFAFA"

DPI = 300
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures_v2", "v25")
os.makedirs(OUT_DIR, exist_ok=True)


def draw_panel_a(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    codons = ["GCU", "GCC", "AAA", "GCU", "UCC"]
    aa = ["Ala", "Ala", "Lys", "Ala", "Ser"]

    y_codon = 5.5
    y_aa = 3.0
    x_start = 1.5
    box_w = 1.2
    box_h = 0.8
    gap = 0.3

    for i, (c, a) in enumerate(zip(codons, aa)):
        x = x_start + i * (box_w + gap)
        color = ORANGE if c == "GCU" and i == 0 else TEAL if c == "GCC" and i == 1 else SLATE
        rect = FancyBboxPatch((x, y_codon), box_w, box_h,
                               boxstyle="round,pad=0.1", facecolor=LIGHT_TEAL,
                               edgecolor=TEAL, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y_codon + box_h / 2, c,
                ha="center", va="center", fontsize=11, fontweight="bold", color=TEAL)

        rect_aa = FancyBboxPatch((x, y_aa), box_w, box_h,
                                  boxstyle="round,pad=0.1", facecolor=LIGHT_ORANGE,
                                  edgecolor=ORANGE, linewidth=1.5)
        ax.add_patch(rect_aa)
        ax.text(x + box_w / 2, y_aa + box_h / 2, a,
                ha="center", va="center", fontsize=11, fontweight="bold", color=ORANGE)

        ax.annotate("", xy=(x + box_w / 2, y_aa + box_h),
                     xytext=(x + box_w / 2, y_codon),
                     arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.2))

    ax.annotate("translation", xy=(x_start + 2 * (box_w + gap) + box_w / 2, y_codon - 0.3),
                xytext=(x_start + 2 * (box_w + gap) + box_w / 2, y_codon - 0.3),
                ha="center", va="top", fontsize=9, fontstyle="italic", color=SLATE)

    ax.annotate("",
                xy=(x_start + 1 * (box_w + gap) + box_w / 2, y_codon + box_h + 0.1),
                xytext=(x_start + 1 * (box_w + gap) + box_w / 2, y_codon + box_h + 0.8),
                arrowprops=dict(arrowstyle="<-", color=TEAL, lw=2.5))
    ax.text(x_start + 1 * (box_w + gap) + box_w / 2, y_codon + box_h + 1.0,
            "σ channel\n(folding, stability)",
            ha="center", va="bottom", fontsize=8, color=TEAL, fontweight="bold")

    ax.text(0.3, 7.5, "CDS", fontsize=12, fontweight="bold", color=TEAL)
    ax.text(0.3, y_codon + box_h / 2, "codons", fontsize=9, color=TEAL)
    ax.text(0.3, y_aa + box_h / 2, "protein", fontsize=9, color=ORANGE)

    plm_x = x_start + 4 * (box_w + gap) + box_w + 0.5
    plm_y = y_aa + box_h / 2
    rect_plm = FancyBboxPatch((plm_x, plm_y - 0.5), 1.8, 1.0,
                               boxstyle="round,pad=0.1", facecolor=LIGHT_ORANGE,
                               edgecolor=ORANGE, linewidth=2)
    ax.add_patch(rect_plm)
    ax.text(plm_x + 0.9, plm_y, "pLM", ha="center", va="center",
            fontsize=10, fontweight="bold", color=ORANGE)
    ax.text(plm_x + 0.9, plm_y - 0.7, "sees A only", ha="center", va="center",
            fontsize=7, color=ORANGE)

    ax.plot([plm_x - 0.3, plm_x], [plm_y + 0.5, plm_y + 0.5], color=ORANGE, lw=1.5)
    ax.plot([plm_x - 0.3, plm_x], [plm_y - 0.3, plm_y - 0.3], color=TEAL, lw=1.5, ls="--")
    ax.text(plm_x - 0.5, plm_y + 0.5, "✓", fontsize=10, color=ORANGE, ha="center", va="center")
    ax.text(plm_x - 0.5, plm_y - 0.3, "✗", fontsize=10, color="red", ha="center", va="center")

    eq_y = 0.8
    ax.text(5, eq_y, r"$I(\mathrm{CDS};Y) = I(A;Y) + I(\sigma;Y|A)$",
            ha="center", va="center", fontsize=13, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE, edgecolor=SLATE, linewidth=1.5))

    ax.text(3.2, eq_y - 0.5, r"$I(A;Y)$", fontsize=9, color=ORANGE, ha="center", fontweight="bold")
    ax.text(7.0, eq_y - 0.5, r"$I(\sigma;Y|A)$", fontsize=9, color=TEAL, ha="center", fontweight="bold")

    ax.text(5, 0.0, "P1: on synonymous variants, $I(A;Y)=0$ → only σ channel carries signal",
            ha="center", va="center", fontsize=7.5, color=SLATE, fontstyle="italic")


def draw_panel_b(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    seq = "GCU GCC AAA GCU UCC"
    codons = seq.split()

    rows = [
        ("codon", "[GCU] [GCC] [AAA] [GCU] [UCC]", TEAL, "✓ aligned", LIGHT_TEAL),
        ("BPE / 6-mer", "[GCU GCC] [AAA GCU] [UCC ...]", SLATE, "~ partial", "#E8E8E8"),
        ("character", "[G][C][U][G][C][C][A][A][A]...", LAVENDER, "✗ broken", "#F0ECF5"),
    ]

    y_positions = [6.0, 3.5, 1.0]
    for idx, (label, tokens_str, color, status, bg_color) in enumerate(rows):
        y = y_positions[idx]

        ax.text(0.2, y + 0.6, label, fontsize=10, fontweight="bold", color=color, va="center")

        rect = FancyBboxPatch((0.1, y - 0.3), 9.5, 1.5,
                               boxstyle="round,pad=0.15", facecolor=bg_color,
                               edgecolor=color, linewidth=1.5, alpha=0.5)
        ax.add_patch(rect)

        if idx == 0:
            token_list = ["GCU", "GCC", "AAA", "GCU", "UCC"]
            x_pos = 1.0
            for t in token_list:
                tw = FancyBboxPatch((x_pos, y - 0.05), 1.1, 0.6,
                                     boxstyle="round,pad=0.08", facecolor=WHITE,
                                     edgecolor=color, linewidth=1.5)
                ax.add_patch(tw)
                ax.text(x_pos + 0.55, y + 0.25, t, ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)
                x_pos += 1.3

            mut_x = 1.0
            ax.add_patch(FancyBboxPatch((mut_x, y - 0.05), 1.1, 0.6,
                                         boxstyle="round,pad=0.08", facecolor="#FFEB3B",
                                         edgecolor="#F44336", linewidth=2.5, alpha=0.5))
            ax.text(mut_x + 0.55, y + 0.25, "GCU", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)
            ax.annotate("1 token\nchanged", xy=(mut_x + 0.55, y - 0.15),
                        fontsize=7, color="#F44336", ha="center", va="top")

        elif idx == 1:
            token_list = ["GCU GCC", "AAA GCU", "UCC ..."]
            x_pos = 1.0
            for t in token_list:
                tw = FancyBboxPatch((x_pos, y - 0.05), 2.0, 0.6,
                                     boxstyle="round,pad=0.08", facecolor=WHITE,
                                     edgecolor=color, linewidth=1.5, linestyle="--")
                ax.add_patch(tw)
                ax.text(x_pos + 1.0, y + 0.25, t, ha="center", va="center",
                        fontsize=8, color=color)
                x_pos += 2.2
            ax.text(1.0, y - 0.25, "codon boundary partially preserved",
                    fontsize=7, color=SLATE, fontstyle="italic")

        elif idx == 2:
            nts = list("GCUGCCAAA")
            x_pos = 1.0
            for i, nt in enumerate(nts):
                tw = FancyBboxPatch((x_pos, y - 0.05), 0.55, 0.6,
                                     boxstyle="round,pad=0.05", facecolor=WHITE,
                                     edgecolor=color, linewidth=1.0)
                ax.add_patch(tw)
                ax.text(x_pos + 0.275, y + 0.25, nt, ha="center", va="center",
                        fontsize=8, color=color)
                if i in [2, 5]:
                    ax.plot([x_pos + 0.55 + 0.02, x_pos + 0.55 + 0.02],
                            [y - 0.05, y + 0.55], color="red", linewidth=1.5,
                            linestyle="--", alpha=0.6)
                x_pos += 0.6
            ax.text(1.0, y - 0.25, "codon boundaries broken; 3 tokens changed per mutation",
                    fontsize=7, color=LAVENDER, fontstyle="italic")

        ax.text(9.3, y + 0.25, status, fontsize=9, fontweight="bold", color=color,
                ha="center", va="center")

    ax.annotate("", xy=(9.7, 6.5), xytext=(9.7, 1.5),
                arrowprops=dict(arrowstyle="<->", color=SLATE, lw=1.5))
    ax.text(9.9, 4.0, "inductive\nbias", fontsize=8, color=SLATE, ha="left", va="center",
            rotation=90, fontstyle="italic")

    ax.text(5, 7.5, "Tokenization as inductive bias", fontsize=11, fontweight="bold",
            color=SLATE, ha="center", va="center")


def draw_panel_c(ax):
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    tiers = [
        ("Tier 0\nZero-shot LLR", 0.50, 0.50, "AUC ≈ 0.5 (random)", "0.5", "0.5"),
        ("Tier 1\nLinear probing (LR)", 0.70, 0.69, "no significant\ndifference (p > 0.5)", "0.70", "0.69"),
        ("Tier 2\nNonlinear probing (MLP)", 0.83, 0.65, "codon advantage\nemerges (p < 0.001)", "0.83", "0.65"),
        ("Tier 3\nLoRA fine-tuning", 0.91, 0.91, "signal fully\naccessible", "0.91", "0.91"),
    ]

    y_positions = [8.5, 6.2, 3.9, 1.6]
    bar_x_codon = 3.5
    bar_x_char = 6.0
    bar_w = 1.8
    max_bar_h = 2.0

    for i, (label, codon_val, char_val, annotation, codon_str, char_str) in enumerate(tiers):
        y = y_positions[i]

        ax.text(1.8, y + 0.5, label, fontsize=9, fontweight="bold",
                color=SLATE, ha="center", va="center")

        codon_h = (codon_val - 0.4) / 0.6 * max_bar_h
        char_h = (char_val - 0.4) / 0.6 * max_bar_h

        rect_c = FancyBboxPatch((bar_x_codon, y - 0.3), bar_w, codon_h,
                                 boxstyle="round,pad=0.05", facecolor=TEAL,
                                 edgecolor=TEAL, linewidth=1.5, alpha=0.85)
        ax.add_patch(rect_c)
        ax.text(bar_x_codon + bar_w / 2, y - 0.3 + codon_h + 0.15, codon_str,
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=TEAL)

        rect_ch = FancyBboxPatch((bar_x_char, y - 0.3), bar_w, char_h,
                                  boxstyle="round,pad=0.05", facecolor=LAVENDER,
                                  edgecolor=LAVENDER, linewidth=1.5, alpha=0.85)
        ax.add_patch(rect_ch)
        ax.text(bar_x_char + bar_w / 2, y - 0.3 + char_h + 0.15, char_str,
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=LAVENDER)

        ax.text(bar_x_codon + bar_w / 2, y - 0.6, "codon", ha="center", va="top",
                fontsize=8, color=TEAL)
        ax.text(bar_x_char + bar_w / 2, y - 0.6, "char", ha="center", va="top",
                fontsize=8, color=LAVENDER)

        ax.text(9.5, y + 0.3, annotation, fontsize=8, color=SLATE, ha="center", va="center")

        if i == 2:
            delta = codon_val - char_val
            ax.annotate("", xy=(bar_x_codon + bar_w / 2, y - 0.3 + codon_h),
                         xytext=(bar_x_char + bar_w / 2, y - 0.3 + char_h),
                         arrowprops=dict(arrowstyle="<->", color="red", lw=2))
            mid_y = y - 0.3 + (codon_h + char_h) / 2
            ax.text(5.5, mid_y + 0.3, f"+{delta:.1%}".replace("%", " pp").replace("0.", ""),
                    fontsize=10, fontweight="bold", color="red", ha="center")

    lock_x = 8.5
    lock_y = (y_positions[1] + y_positions[2]) / 2 + 0.3
    rect_lock = FancyBboxPatch((lock_x - 0.8, lock_y - 0.4), 1.6, 0.8,
                                boxstyle="round,pad=0.1", facecolor="#FFF3E0",
                                edgecolor=ORANGE, linewidth=2)
    ax.add_patch(rect_lock)
    ax.text(lock_x, lock_y, "LOCKED → UNLOCKED", fontsize=8, fontweight="bold",
            ha="center", va="center", color=ORANGE)
    ax.text(lock_x, lock_y - 0.6, "linear probing\nlocks the signal;\nMLP unlocks it",
            fontsize=7, ha="center", va="top", color=SLATE, fontstyle="italic")

    ax.annotate("", xy=(0.5, 1.0), xytext=(0.5, 9.0),
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=2))
    ax.text(0.3, 5.0, "extractable\ndepth", fontsize=9, fontweight="bold",
            color=SLATE, ha="center", va="center", rotation=90)

    missense_x = 12.5
    ax.plot([missense_x - 0.5, missense_x + 0.5],
            [y_positions[1] + 0.5, y_positions[1] + 0.5],
            color=ORANGE, lw=3, solid_capstyle="round")
    ax.text(missense_x, y_positions[1] + 0.9, "missense\n(readable at Tier 1)",
            fontsize=7, color=ORANGE, ha="center", va="bottom", fontweight="bold")

    syn_y = [y_positions[0] + 0.5, y_positions[1] + 0.5, y_positions[2] + 0.5, y_positions[3] + 0.5]
    ax.plot([missense_x - 0.5, missense_x - 0.5, missense_x + 0.5, missense_x + 0.5],
            [syn_y[0], syn_y[1], syn_y[2], syn_y[3]],
            color=TEAL, lw=3, solid_capstyle="round")
    ax.text(missense_x, y_positions[3] + 0.9, "synonymous\n(emerges at Tier 2)",
            fontsize=7, color=TEAL, ha="center", va="bottom", fontweight="bold")

    ax.text(7, 9.6, "The synonymous channel carries information at nonlinear depth",
            fontsize=11, fontweight="bold", color=SLATE, ha="center", va="center")

    ax.text(7, 0.2,
            "Outeiral & Deane [9] showed codon > amino-acid; we explain why: the advantage lives at nonlinear depth",
            fontsize=7.5, color=SLATE, ha="center", va="center", fontstyle="italic")


def draw_panel_d(ax):
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(7, 5.5, "CodonBench Framework (hand-drawing needed for this panel)",
            fontsize=11, fontweight="bold", color=SLATE, ha="center", va="center")

    sections = [
        (1.5, "INPUTS\n(21 models)", [
            ("cLMs ×9", TEAL),
            ("char-CDS ×2", LAVENDER),
            ("pLMs ×2", ORANGE),
            ("DNA LMs ×2", SLATE),
            ("non-neural ×6", "#999999"),
        ]),
        (5.5, "4-TIER\nPROBING", [
            ("Tier 0: LLR", SLATE),
            ("Tier 1: LR", ORANGE),
            ("Tier 2: MLP", TEAL),
            ("Tier 3: LoRA", TEAL),
        ]),
        (9.5, "TASKS\n(6 tasks)", [
            ("SynPath (σ)", TEAL),
            ("MisPath (A)", ORANGE),
            ("Expression ×4", SLATE),
        ]),
    ]

    for x_center, title, items in sections:
        rect = FancyBboxPatch((x_center - 1.5, 0.5), 3.0, 4.5,
                               boxstyle="round,pad=0.2", facecolor=WHITE,
                               edgecolor=SLATE, linewidth=1.5, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x_center, 4.5, title, fontsize=9, fontweight="bold",
                color=SLATE, ha="center", va="center")
        for j, (item, color) in enumerate(items):
            y_item = 3.5 - j * 0.7
            ax.text(x_center, y_item, item, fontsize=8, color=color,
                    ha="center", va="center")

    for x_from, x_to in [(3.0, 4.0), (7.0, 8.0)]:
        ax.annotate("", xy=(x_to, 2.5), xytext=(x_from, 2.5),
                     arrowprops=dict(arrowstyle="->", color=SLATE, lw=2))

    output_x = 12.5
    rect_out = FancyBboxPatch((output_x - 1.2, 1.0), 2.4, 3.0,
                               boxstyle="round,pad=0.2", facecolor=WHITE,
                               edgecolor=SLATE, linewidth=1.5, alpha=0.8)
    ax.add_patch(rect_out)
    ax.text(output_x, 3.5, "OUTPUT", fontsize=9, fontweight="bold",
            color=SLATE, ha="center", va="center")
    ax.text(output_x, 2.7, "σ: ρ = 0.59\n(rankings reorder)",
            fontsize=8, color=TEAL, ha="center", va="center")
    ax.text(output_x, 1.7, "A: ρ = 0.86\n(rankings stable)",
            fontsize=8, color=ORANGE, ha="center", va="center")

    ax.annotate("", xy=(11.3, 2.5), xytext=(10.5, 2.5),
                 arrowprops=dict(arrowstyle="->", color=SLATE, lw=2))


def main():
    fig = plt.figure(figsize=(18, 22), facecolor=WHITE)

    ax_a = fig.add_axes([0.05, 0.72, 0.42, 0.25])
    ax_b = fig.add_axes([0.53, 0.72, 0.42, 0.25])
    ax_c = fig.add_axes([0.05, 0.28, 0.90, 0.40])
    ax_d = fig.add_axes([0.05, 0.02, 0.90, 0.22])

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)
    draw_panel_d(ax_d)

    ax_a.text(-0.02, 1.02, "a", transform=ax_a.transAxes,
              fontsize=18, fontweight="bold", va="top")
    ax_b.text(-0.02, 1.02, "b", transform=ax_b.transAxes,
              fontsize=18, fontweight="bold", va="top")
    ax_c.text(-0.01, 1.02, "c", transform=ax_c.transAxes,
              fontsize=18, fontweight="bold", va="top")
    ax_d.text(-0.01, 1.02, "d", transform=ax_d.transAxes,
              fontsize=18, fontweight="bold", va="top")

    for fmt in ["png", "pdf"]:
        path = os.path.join(OUT_DIR, f"fig1_framework.{fmt}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=WHITE)
        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()