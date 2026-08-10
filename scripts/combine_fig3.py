from PIL import Image
import os

base = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\figures_v2\v26'

imgs = {}
for key, fn in [('a', 'fig3a_lr_to_mlp_slope.png'), ('b', 'fig3b_protocol_sensitivity.png'), ('c', 'fig3c_production_mlp_auc.png')]:
    path = os.path.join(base, fn)
    if os.path.exists(path):
        im = Image.open(path)
        im.thumbnail((1400, 700), Image.LANCZOS)
        imgs[key] = im
    else:
        print(f'MISSING: {fn}')

wa, ha = imgs['a'].size
wb, hb = imgs['b'].size
wc, hc = imgs['c'].size

pad = 15
total_w = wa + pad + wb + pad + wc
total_h = max(ha, hb, hc)

canvas = Image.new('RGB', (total_w + pad * 2, total_h + pad * 2), 'white')

def paste_centered(img, box_x, box_y, box_w, box_h):
    iw, ih = img.size
    offset_x = box_x + (box_w - iw) // 2
    offset_y = box_y + (box_h - ih) // 2
    canvas.paste(img, (offset_x, offset_y))

x = pad
paste_centered(imgs['a'], x, pad, wa, total_h)
x += wa + pad
paste_centered(imgs['b'], x, pad, wb, total_h)
x += wb + pad
paste_centered(imgs['c'], x, pad, wc, total_h)

out = os.path.join(base, 'fig3_combined.png')
canvas.save(out, dpi=(150, 150))
print(f'Done: {out} ({canvas.size})')