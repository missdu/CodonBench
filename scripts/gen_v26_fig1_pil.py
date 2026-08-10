"""Combine Fig 1a, 1b, 1c into a single Fig 1 using PIL.
Layout: top row = a (left) + b (right), bottom row = c (full width).
"""
from PIL import Image
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE, 'paper', 'figures_v2', 'v26')

def combine_fig1():
    img_a = Image.open(os.path.join(FIG_DIR, 'fig1a_schematic.png'))
    img_b = Image.open(os.path.join(FIG_DIR, 'fig1b_flow.png'))
    img_c = Image.open(os.path.join(FIG_DIR, 'fig2_forest_compact.png'))

    w_a, h_a = img_a.size
    w_b, h_b = img_b.size
    w_c, h_c = img_c.size

    # Top row: a + b side by side, aligned at top
    top_h = max(h_a, h_b)
    top_w = w_a + w_b + 40  # 40px gap

    # Scale c to match top row width
    scale = top_w / w_c
    new_w_c = int(w_c * scale)
    new_h_c = int(h_c * scale)
    img_c_scaled = img_c.resize((new_w_c, new_h_c), Image.LANCZOS)

    # Total canvas
    total_w = top_w
    total_h = top_h + 40 + new_h_c  # 40px gap between rows

    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))

    # Paste a (left-aligned, top-aligned)
    canvas.paste(img_a, (0, 0))

    # Paste b (right of a, top-aligned)
    canvas.paste(img_b, (w_a + 40, 0))

    # Paste c (centered, below top row)
    c_x = (total_w - new_w_c) // 2
    canvas.paste(img_c_scaled, (c_x, top_h + 40))

    out_path = os.path.join(FIG_DIR, 'fig1_combined.png')
    canvas.save(out_path, dpi=(300, 300))
    print(f"Saved to {out_path} ({total_w}x{total_h})")

    # Also save PDF
    out_pdf = os.path.join(FIG_DIR, 'fig1_combined.pdf')
    canvas.save(out_pdf, 'PDF', resolution=300)
    print(f"Saved to {out_pdf}")

if __name__ == '__main__':
    combine_fig1()