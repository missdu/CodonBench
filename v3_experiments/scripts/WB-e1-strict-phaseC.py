# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB | E1 严格版 —— Phase C：配对检验

Phase B 显示 codonbert 的 diff 臂在 LOGO 下 AUC 0.5696，
而假变异对照 pseudo_diff 是 0.4844。两者 CI 不重叠，但**没有做配对比较**。

本脚本补上：
  对同一批折、同一批样本，计算 Δ = AUC(diff) − AUC(pseudo_diff)，
  用按基因聚类的 bootstrap 给出 Δ 的 95% CI。
  同时给出 DeLong 近似的逐折 Δ 均值 ± 标准误。

这是"cLM 的变异响应向量是否携带可迁移信号"这一声称的直接证据。
"""
import os
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score

BASE = Path(os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp"))
OUT = BASE / "results" / "supplementary" / "wb_rerun"

usable = np.load(OUT / "e1s_usable.npy")
pse_all = list(np.load(OUT / "e1s_pseudo.npy", allow_pickle=True))
pse_seqs = [pse_all[i] for i in range(len(pse_all)) if usable[i]]
labels = np.load(OUT / "e1s_labels.npy")[usable].astype(int)
genes_all = list(np.load(OUT / "e1s_genes.npy", allow_pickle=True))
genes = np.array([genes_all[i] for i in range(len(usable)) if usable[i]], dtype=object)
n = len(labels)
print("样本 %d | 基因 %d" % (n, len(set(genes))))

E_ref = np.load(OUT / "e1s_codonbert_ref.npy")
E_alt = np.load(OUT / "e1s_codonbert_alt.npy")
print("已存嵌入 ref %s alt %s" % (E_ref.shape, E_alt.shape))

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def to_rna_codons(x):
    r = x.replace("T", "U")
    return " ".join([r[i:i + 3] for i in range(0, len(r) - len(r) % 3, 3)])


tok = AutoTokenizer.from_pretrained("lhallee/CodonBERT", trust_remote_code=True)
mdl = AutoModel.from_pretrained("lhallee/CodonBERT", trust_remote_code=True).to(device).eval()


@torch.no_grad()
def embed(seqs, bs=32):
    out = []
    for i in range(0, len(seqs), bs):
        batch = [to_rna_codons(s) for s in seqs[i:i + bs]]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(device)
        h = mdl(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out, 0)


E_pse = embed(pse_seqs)
np.save(OUT / "e1s_codonbert_pseudo.npy", E_pse)
print("pseudo 嵌入 %s" % (E_pse.shape,))

Xdiff = E_alt - E_ref
Xpse = E_pse - E_ref

res = {}
for split_name in ["random", "LOGO"]:
    if split_name == "random":
        folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(Xdiff, labels))
    else:
        folds = list(GroupKFold(5).split(Xdiff, labels, genes))
    oof_d = np.zeros(n)
    oof_p = np.zeros(n)
    fold_auc_d, fold_auc_p = [], []
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(Xdiff[tr], labels[tr])
        oof_d[te] = clf.predict_proba(Xdiff[te])[:, 1]
        clf2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf2.fit(Xpse[tr], labels[tr])
        oof_p[te] = clf2.predict_proba(Xpse[te])[:, 1]
        if len(np.unique(labels[te])) == 2:
            fold_auc_d.append(roc_auc_score(labels[te], oof_d[te]))
            fold_auc_p.append(roc_auc_score(labels[te], oof_p[te]))
    a_d = roc_auc_score(labels, oof_d)
    a_p = roc_auc_score(labels, oof_p)

    # 逐折 Δ
    fd = np.array(fold_auc_d)
    fp = np.array(fold_auc_p)
    d_fold = fd - fp
    print("\n[%s] 逐折 AUC  diff %s" % (split_name, np.round(fd, 4)))
    print("      逐折 AUC pseudo %s" % np.round(fp, 4))
    print("      逐折 Δ %s | 均值 %.4f ± %.4f (SEM)"
          % (np.round(d_fold, 4), d_fold.mean(), d_fold.std(ddof=1) / np.sqrt(len(d_fold))))

    # 按基因聚类 bootstrap 的 Δ
    rng = np.random.RandomState(7)
    ug = np.unique(genes)
    gi = {g: np.where(genes == g)[0] for g in ug}
    boots = []
    for _ in range(1000):
        pick = rng.choice(len(ug), size=len(ug), replace=True)
        idx = np.concatenate([gi[ug[p]] for p in pick])
        if len(np.unique(labels[idx])) < 2:
            continue
        try:
            boots.append(roc_auc_score(labels[idx], oof_d[idx])
                         - roc_auc_score(labels[idx], oof_p[idx]))
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print("      Δ(diff − pseudo) = %.4f  95%%CI [%.4f, %.4f]" % (a_d - a_p, lo, hi))
    print("      ⇒ CI 不含 0 即说明：变异响应向量的预测力显著高于"
          "「同一窗口随便换个碱基」的响应")
    res[split_name] = {
        "auc_diff": round(float(a_d), 4),
        "auc_pseudo": round(float(a_p), 4),
        "delta": round(float(a_d - a_p), 4),
        "delta_ci_lo": round(float(lo), 4),
        "delta_ci_hi": round(float(hi), 4),
        "fold_auc_diff": [round(float(x), 4) for x in fd],
        "fold_auc_pseudo": [round(float(x), 4) for x in fp],
        "fold_delta_mean": round(float(d_fold.mean()), 4),
        "fold_delta_sem": round(float(d_fold.std(ddof=1) / np.sqrt(len(d_fold))), 4),
    }

with open(OUT / "e1_strict_paired.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
print("\n已存:", OUT / "e1_strict_paired.json")
