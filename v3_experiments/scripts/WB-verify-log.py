#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WB-verify-log.py —— 实验日志数字对账

用途
    把 `WB-实验日志.md` 表格里写的每一个数字，拿去和 `data/*.json` 的原始字段逐条比对。
    改过日志或改过数据后跑一遍，防止"日志里的数"和"文件里的数"分叉。

用法
    python WB-verify-log.py            # 在 WB-out/ 下直接跑
    python WB-verify-log.py --log X.md --data DIR

判据
    容差 6e-4（日志里的数四舍五入到 3 位小数，这是唯一允许的偏差来源）。
    任何一个数超差 ⇒ 打印出来并 exit(1)。

注意
    日志表格必须用 **ASCII 减号** `-`，不要用 U+2212 `−`，否则正则匹配不到。
    脚本会自动检测 U+2212 并给出提示。
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 6e-4

# 显示名 -> JSON 里的键
MODELS = {
    'CodonBERT': 'codonbert',
    'CodonBERT-HF': 'codonbert_hf',
    'EnCodon-80M': 'encodon-80m',
    'mRNABERT': 'mrnabert',
    'ESM-2-650M': 'ESM-2-650M',
    'ESM-1b-650M': 'ESM-1b-650M',
}
# JSON 键 -> 显示名（B-5 的 paired 字段名是小写）
DISP = {
    'codonbert': 'CodonBERT',
    'codonbert_hf': 'CodonBERT-HF',
    'encodon-80m': 'EnCodon-80M',
    'mrnabert': 'mRNABERT',
}

bad = []


def chk(tag, written, actual, tol=TOL):
    """written = 日志里写的数；actual = JSON 里的数"""
    if actual is None:
        bad.append((tag, written, '字段缺失'))
        return
    if abs(written - actual) > tol:
        bad.append((tag, written, actual))


def row(pattern, log, tag):
    m = re.search(pattern, log, re.M)
    if not m:
        bad.append((tag, '行未匹配', None))
    return m


def load(data_dir, name):
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        bad.append((name, '文件缺失', path))
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def main(log_path, data_dir):
    with open(log_path, encoding='utf-8') as f:
        log = f.read()

    if '\u2212' in log:
        print('[警告] 日志里还有 U+2212（Unicode 减号），会导致本脚本匹配不到。')
        print('       执行：python -c "p=\'%s\';s=open(p,encoding=\'utf-8\').read();'
              'open(p,\'w\',encoding=\'utf-8\').write(s.replace(chr(0x2212),\'-\'))"' % log_path)

    # ---- EXP-001 E2 ----
    e2 = load(data_dir, 'e2_rerun_WB.json')
    if e2:
        for nm, k in MODELS.items():
            v = e2['models'][k]
            m = row(r'^\| %s \| ([\d.]+) \| ([\d.]+) \| ([+-][\d.]+) \| ([+-][\d.]+) \|'
                    % re.escape(nm), log, 'E2.' + nm)
            if m:
                chk('E2.%s.G1' % nm, float(m.group(1)), v['G1_known_gene']['auc_lr'])
                chk('E2.%s.G2' % nm, float(m.group(2)), v['G2_unseen_gene']['auc_lr'])
                chk('E2.%s.dG1' % nm, float(m.group(3)), v['delta_add']['G1_known_gene'])
                chk('E2.%s.dG2' % nm, float(m.group(4)), v['delta_add']['G2_unseen_gene'])
        chk('E2.bg.G1', 0.611, e2['baseline']['bg_G1_known_gene']['auc_lr'])
        chk('E2.bg.G2', 0.598, e2['baseline']['bg_G2_unseen_gene']['auc_lr'])

    # ---- EXP-002 B-5 ----
    b5 = load(data_dir, 'b5_rerun_WB.json')
    if b5:
        for nm, k in MODELS.items():
            m = row(r'^\| %s \| ([\d.]+) \|$' % re.escape(nm), log, 'B5.' + nm)
            if m:
                chk('B5.%s' % nm, float(m.group(1)), b5['models'][k]['auc_pooled'])
        for key, v in b5['paired'].items():
            a, b = key.split(' - ')
            m = row(r'^\| %s - %s \| ([+-][\d.]+) \| \[([+-][\d.]+), ([+-][\d.]+)\] \|'
                    % (re.escape(DISP[a]), re.escape(b)), log, 'B5.' + key)
            if m:
                chk('B5.%s.d' % key, float(m.group(1)), v['point_diff'])
                chk('B5.%s.lo' % key, float(m.group(2)), v['ci95_lo'])
                chk('B5.%s.hi' % key, float(m.group(3)), v['ci95_hi'])

    # ---- EXP-003 E3 ----
    e3 = load(data_dir, 'e3_rerun_WB.json')
    if e3:
        for nm, k in MODELS.items():
            v = e3['models'][k]
            m = row(r'^\| \*{0,2}%s\*{0,2} \| ([\d.]+) \| ([\d.]+) \| \*{0,2}([+-][\d.]+)\*{0,2} \|'
                    r' \[([+-][\d.]+), ([+-][\d.]+)\] \|' % re.escape(nm), log, 'E3.' + nm)
            if m:
                chk('E3.%s.support' % nm, float(m.group(1)), v['auc_support'])
                chk('E3.%s.exclude' % nm, float(m.group(2)), v['auc_exclude'])
                chk('E3.%s.delta' % nm, float(m.group(3)), v['delta'])
                chk('E3.%s.lo' % nm, float(m.group(4)), v['ci95_lo'])
                chk('E3.%s.hi' % nm, float(m.group(5)), v['ci95_hi'])

    # ---- EXP-004 E1 ----
    e1 = load(data_dir, 'e1_strict_results.json')
    p1 = load(data_dir, 'e1_strict_paired.json')
    if e1:
        arms = [('ref_random', 0.652), ('alt_random', 0.669), ('diff_random', 0.718),
                ('pair_random', 0.755), ('pseudo_diff_random', 0.583),
                ('ref_LOGO', 0.501), ('alt_LOGO', 0.518), ('diff_LOGO', 0.570),
                ('pair_LOGO', 0.559), ('pseudo_diff_LOGO', 0.484)]
        for f, exp in arms:
            chk('E1.codonbert.%s' % f, exp, e1['codonbert'][f]['auc'])
        esm = e1['esm1b-650m']
        chk('E1.esm1b.ref_random', 0.736, esm['ref_random']['auc'])
        chk('E1.esm1b.ref_LOGO', 0.544, esm['ref_LOGO']['auc'])
        chk('E1.esm1b.diff_random', 0.499, esm['diff_random']['auc'])
        chk('E1.esm1b.diff_LOGO', 0.479, esm['diff_LOGO']['auc'])
        # 两条自检：cLM 必须 != 0，pLM 必须 == 0（恒等式）
        chk('E1.selfcheck.clm', 0.0392, e1['codonbert']['_selfcheck']['mean_abs_ref_alt'], 1e-4)
        if esm['_selfcheck']['mean_abs_ref_alt'] != 0.0:
            bad.append(('E1.selfcheck.plm', '必须为 0（恒等式）',
                        esm['_selfcheck']['mean_abs_ref_alt']))
        chk('E1.perm.random', 0.482, e1['codonbert']['_permuted']['diff_random']['auc'])
        chk('E1.perm.LOGO', 0.478, e1['codonbert']['_permuted']['diff_LOGO']['auc'])
    if p1:
        for sp, exp in [('random', (0.718, 0.583, 0.135, 0.106, 0.166)),
                        ('LOGO', (0.570, 0.484, 0.085, 0.051, 0.124))]:
            v = p1[sp]
            chk('E1.paired.%s.diff' % sp, exp[0], v['auc_diff'])
            chk('E1.paired.%s.pseudo' % sp, exp[1], v['auc_pseudo'])
            chk('E1.paired.%s.delta' % sp, exp[2], v['delta'])
            chk('E1.paired.%s.lo' % sp, exp[3], v['delta_ci_lo'])
            chk('E1.paired.%s.hi' % sp, exp[4], v['delta_ci_hi'])
        chk('E1.paired.random.foldmean', 0.135, p1['random']['fold_delta_mean'])
        chk('E1.paired.LOGO.foldmean', 0.086, p1['LOGO']['fold_delta_mean'])
        chk('E1.paired.random.sem', 0.014, p1['random']['fold_delta_sem'])
        chk('E1.paired.LOGO.sem', 0.012, p1['LOGO']['fold_delta_sem'])

    # ---- EXP-005 G3 ----
    g3 = load(data_dir, 'g3_rerun_WB.json')
    if g3:
        S = g3['summary']['models']
        for nm, k in MODELS.items():
            v = S[k]
            c = v['G3_minus_ctrl']
            m = row(r'^\| %s \| ([\d.]+) \| ([\d.]+) \| \*\*([\d.]+)\*\* \| ([\d.]+) \|'
                    r' ([+-][\d.]+) \xb1 ([\d.]+) \|' % re.escape(nm), log, 'G3.' + nm)
            if m:
                chk('G3.%s.G1' % nm, float(m.group(1)), v['G1']['mean'])
                chk('G3.%s.G2' % nm, float(m.group(2)), v['G2']['mean'])
                chk('G3.%s.G3' % nm, float(m.group(3)), v['G3']['mean'])
                chk('G3.%s.ctrl' % nm, float(m.group(4)), c['ctrl_mean'])
                chk('G3.%s.diff' % nm, float(m.group(5)), c['mean_diff'])
                chk('G3.%s.sd' % nm, float(m.group(6)), c['sd_diff'])
        b = g3['summary']['baseline']
        ctrl = (b['G3ctrl_s0']['mean'] + b['G3ctrl_s1']['mean'] + b['G3ctrl_s2']['mean']) / 3
        chk('G3.bg.G1', 0.607, b['G1']['mean'])
        chk('G3.bg.G2', 0.600, b['G2']['mean'])
        chk('G3.bg.G3', 0.632, b['G3']['mean'])
        chk('G3.bg.ctrl', 0.605, ctrl)
        chk('G3.bg.diff', 0.027, b['G3']['mean'] - ctrl)

    if bad:
        print('[FAIL] %d 处不一致：' % len(bad))
        for t, w, a in bad:
            print('   %-28s 日志=%s  文件=%s' % (t, w, a))
        return 1
    print('[OK] 日志与数据文件逐条一致（容差 %g）' % TOL)
    return 0


if __name__ == '__main__':
    args = sys.argv[1:]
    logp = os.path.join(HERE, 'WB-实验日志.md')
    datad = os.path.join(HERE, 'data')
    if '--log' in args:
        logp = args[args.index('--log') + 1]
    if '--data' in args:
        datad = args[args.index('--data') + 1]
    sys.exit(main(logp, datad))
