#!/usr/bin/env python
# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
import os  # [脱敏] 供读取 CODONBENCH_* 环境变量
"""
WB-诊断-P0-o：CodonBERT 主结果锚点不一致，定位根因并判定是否需要重跑。

不提嵌入、不占 GPU。只做三件事：
  1) 回查所有历史 json 里 CodonBERT 的 Δ_alt 原始记录（不凭日志）
  2) 用磁盘现存各批嵌入分别重算，看哪一格能复现 +0.085
  3) 分折口径对比：GroupKFold(5) vs 逐基因留一（V1 官方 240 折口径）

评估函数逐字复用 EXP-007b / EXP-010。
"""
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

BASE = Path(os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp"))
OUT = BASE / 'results/supplementary/wb_rerun'

FILES = ['e1_strict_results.json', 'e1_strict_paired.json',
         'e1_strict_phaseA_report.json', 'e1_multimodel_results.json',
         'exp010_reviewstatus_results.json']


def dump_all(o, pre=''):
    """把嵌套 json 摊平成 path=value"""
    if isinstance(o, dict):
        for k, v in o.items():
            yield from dump_all(v, pre + '/' + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from dump_all(v, pre + '[' + str(i) + ']')
    else:
        yield pre, o


print('=' * 78)
print('第 1 步：回查历史 json 里 CodonBERT / Δ_alt 的每一条原始记录')
print('=' * 78)
for fn in FILES:
    p = OUT / fn
    if not p.exists():
        print('[缺失]', fn)
        continue
    try:
        d = json.load(open(p))
    except Exception as e:
        print('[解析失败]', fn, e)
        continue
    print('\n---', fn)
    hit = 0
    for path, v in dump_all(d):
        low = (path + '=' + str(v)).lower()
        if any(kw in low for kw in ['codonbert', 'delta', 'paired']):
            print('   %-70s = %s' % (path[:70], v))
            hit += 1
            if hit > 16:
                print('   ... (截断)')
                break

# ---------------------------------------------------------------- 评估函数
def oof_scores(X, y, groups, split):
    folds = (list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
             if split == 'random' else
             list(GroupKFold(5).split(X, y, groups)))
    oof = np.zeros(len(y))
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.decision_function(X[te])
    return oof


def auc_ci(y, s, groups, n_boot=1000, seed=7):
    rng = np.random.RandomState(seed)
    g = np.unique(groups)
    base = roc_auc_score(y, s)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(g, len(g), replace=True)
        idx = np.concatenate([np.where(groups == x)[0] for x in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], s[idx]))
    return base, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def paired_delta(y, groups, Xa, Xb, split, n_boot=1000, seed=11):
    oa = oof_scores(Xa, y, groups, split)
    ob = oof_scores(Xb, y, groups, split)
    rng = np.random.RandomState(seed)
    g = np.unique(groups)
    delta = roc_auc_score(y, oa) - roc_auc_score(y, ob)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(g, len(g), replace=True)
        idx = np.concatenate([np.where(groups == x)[0] for x in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], oa[idx]) - roc_auc_score(y[idx], ob[idx]))
    return {'delta': float(delta),
            'ci_lo': float(np.percentile(boots, 2.5)),
            'ci_hi': float(np.percentile(boots, 97.5)),
            'auc_a': float(roc_auc_score(y, oa)),
            'auc_b': float(roc_auc_score(y, ob))}


def logo_leaveone(X, y, genes, max_folds=None):
    """逐基因留一口径（V1 官方 240 折）：每个基因作一折"""
    g = np.unique(genes)
    if max_folds:
        g = g[:max_folds]
    oof = np.zeros(len(y))
    for gg in g:
        te = np.where(genes == gg)[0]
        tr = np.where(genes != gg)[0]
        if len(np.unique(y[tr])) < 2 or len(te) == 0:
            oof[te] = 0.0
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.decision_function(X[te])
    return oof

# ---------------------------------------------------------------- 载入
u = np.load(OUT / 'e1s_usable.npy').astype(bool)
y = np.load(OUT / 'e1s_labels.npy')
genes = np.load(OUT / 'e1s_genes.npy', allow_pickle=True).astype(str)
y, genes = y[u], genes[u]

batches = {}
# 批次 A：14:14（已知 bug，ref==alt）
if (OUT / 'e1_emb_codonbert_ref.npy').exists():
    batches['A_14:14_bug'] = (
        np.load(OUT / 'e1_emb_codonbert_ref.npy')[u],
        np.load(OUT / 'e1_emb_codonbert_alt.npy')[u])
# 批次 B：14:18 修正批
if (OUT / 'e1v2_codonbert_ref.npy').exists():
    batches['B_14:18_e1v2'] = (
        np.load(OUT / 'e1v2_codonbert_ref.npy')[u],
        np.load(OUT / 'e1v2_codonbert_alt.npy')[u])
# 批次 C：20:32 当前磁盘
batches['C_20:32_e1s'] = (np.load(OUT / 'e1s_codonbert_ref.npy'),
                          np.load(OUT / 'e1s_codonbert_alt.npy'))
PSE = np.load(OUT / 'e1s_codonbert_pse.npy')

print()
print('=' * 78)
print('第 2 步：各批嵌入重算（GroupKFold(5)，配对增量＝AUC(Δ)−AUC(Δ_pse)）')
print('=' * 78)
print('%-16s %8s | %-22s | %-22s' % ('批次', 'mean|Δ|', 'LOGO', 'random'))
for name, (R, A) in batches.items():
    d_ref = A - R
    p_ref = PSE - R
    L = paired_delta(y, genes, d_ref, p_ref, 'LOGO')
    Rn = paired_delta(y, genes, d_ref, p_ref, 'random')
    print('%-16s %8.5f | %+7.4f [%5.3f,%5.3f] | %+7.4f [%5.3f,%5.3f]' % (
        name, np.abs(d_ref).mean(),
        L['delta'], L['ci_lo'], L['ci_hi'],
        Rn['delta'], Rn['ci_lo'], Rn['ci_hi']))

print()
print('=' * 78)
print('第 3 步：分折口径对比（用批次 C）')
print('=' * 78)
R, A = batches['C_20:32_e1s']
d_ref, p_ref = A - R, PSE - R
# 3a GroupKFold(5)
L5 = paired_delta(y, genes, d_ref, p_ref, 'LOGO')
# 3b 逐基因留一
oa = logo_leaveone(d_ref, y, genes)
ob = logo_leaveone(p_ref, y, genes)
L240 = {'auc_a': roc_auc_score(y, oa), 'auc_b': roc_auc_score(y, ob),
        'delta': roc_auc_score(y, oa) - roc_auc_score(y, ob)}
print('GroupKFold(5)      : AUC(Δ)=%.4f  AUC(Δ_pse)=%.4f  Δ=%+.4f [%+.3f,%+.3f]'
      % (L5['auc_a'], L5['auc_b'], L5['delta'], L5['ci_lo'], L5['ci_hi']))
print('逐基因留一(%d 折)  : AUC(Δ)=%.4f  AUC(Δ_pse)=%.4f  Δ=%+.4f  <无 bootstrap CI>'
      % (len(np.unique(genes)), L240['auc_a'], L240['auc_b'], L240['delta']))
print()
print('基因数 =', len(np.unique(genes)), '| 样本 n =', len(y))

print()
print('=' * 78)
print('判定')
print('=' * 78)
print('若第 2 步中某批次重算 ≈ +0.085 ⇒ 该批次即 +0.085 的来源，P0-o 根因确认。')
print('若全部 ≈ +0.069 ⇒ +0.085 不可复现，须以重算值为准。')
print('若第 3 步两口径差异 ≫ 3e-3 ⇒ 根因是分折口径，须统一到 V1 的 240 折。')
