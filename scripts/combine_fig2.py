from PIL import Image
import os

base = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\figures_v2\v26'

panels = {
    'a': 'fig2a_ablation_design.png',
    'b': 'fig2bh_synpath_mispath.png',
    'c': 'fig_layerwise_analysis.png',
    'd': 'fig_extractable_depth_codon_char.png',
    'e': 'fig2d_scale_amplification.png',
    'f': 'fig2f_gap_amplification.png',
}

imgs = {}
for key, fn in panels.items():
    path = os.path.join(base, fn)
    if os.path.exists(path):
        im = Image.open(path)
        im.thumbnail((1200, 1200), Image.LANCZOS)
        imgs[key] = im
    else:
        print(f'MISSING: {fn}')

def get_size(key):
    return imgs[key].size

w_left = max(get_size('a')[0], get_size('b')[0], get_size('c')[0])
w_right = max(get_size('d')[0], get_size('e')[0], get_size('f')[0])

h_row0 = get_size('a')[1]
h_row1 = get_size('b')[1]
h_row2 = get_size('c')[1]

pad = 10
total_w = w_left + w_right + pad * 3
total_h = h_row0 + h_row1 + h_row2 + pad * 4

canvas = Image.new('RGB', (total_w, total_h), 'white')

def paste_centered(img, box_x, box_y, box_w, box_h):
    iw, ih = img.size
    offset_x = box_x + (box_w - iw) // 2
    offset_y = box_y + (box_h - ih) // 2
    canvas.paste(img, (offset_x, offset_y))

y = pad
paste_centered(imgs['a'], pad, y, w_left, h_row0)
paste_centered(imgs['d'], w_left + pad * 2, y, w_right, h_row0)
y += h_row0 + pad
paste_centered(imgs['b'], pad, y, w_left, h_row1)
paste_centered(imgs['e'], w_left + pad * 2, y, w_right, h_row1)
y += h_row1 + pad
paste_centered(imgs['c'], pad, y, w_left, h_row2)
paste_centered(imgs['f'], w_left + pad * 2, y, w_right, h_row2)

out = os.path.join(base, 'fig2_combined.png')
canvas.save(out, dpi=(150, 150))
print(f'Done: {out} ({canvas.size})')
