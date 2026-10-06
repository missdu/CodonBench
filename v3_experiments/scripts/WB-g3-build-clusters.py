"""
WB | G3 —— 第 2 步：由 HGNC 家族构造基因簇，并检查规模

要决定的事：
  用哪些家族？HGNC 的 gene_group 里混着两类：
    -  tight：真正的基因家族（BBSome、Fibrillar collagens、Dynein heavy chains）
    -  loose：按结构域/功能打的粗标签（SH3 domain containing、MicroRNA host genes）
  若全用，一个测试基因的"家族"可能牵连掉大半训练集 ⇒ G3 就没法训练了。

做法：两套定义都算出来，看规模再定主次
  DEF-A (all)  ：任一共享家族即视为同源
  DEF-B (tight)：只用在**全基因组层面**也确实是"小家族"的标签
                 —— 这里用本数据集内该家族的基因数 ≤ 10 作为判据
  DEF-C (specific)：每个基因取其**最特异**（本数据集中最小）的家族作为唯一归属
                 —— 这样每个基因恰好属于一个簇，便于做分组 CV

输出：g3_gene_clusters.json（基因 -> 各定义下的簇 ID）
"""
import json
import csv
from collections import defaultdict, Counter

HERE = __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0]
HG = json.load(open(HERE + "/data/hgnc_gene_groups.json", encoding="utf-8"))
GENE_CSV = HERE + "/data/g3_gene_list.csv"

by_hgnc = HG["by_hgnc_id"]
gene_to_hgnc = HG["gene_to_hgnc"]
rows = list(csv.DictReader(open(GENE_CSV, encoding="utf-8")))
genes = sorted({r["gene_symbol"] for r in rows})
print("数据集基因数:", len(genes))

# 基因 -> 家族列表
g2groups = {}
for g in genes:
    h = gene_to_hgnc.get(g)
    v = by_hgnc.get(h)
    g2groups[g] = v["gene_group"] if v else []
has = sum(1 for g in genes if g2groups[g])
print("有家族注释:", has, "/", len(genes))

# 家族在本数据集中的规模
gsize = Counter()
for g, gs in g2groups.items():
    for x in gs:
        gsize[x] += 1
print("家族总数:", len(gsize))
print("家族规模分布: 含1个基因 %d | 2-5 %d | 6-10 %d | 11-20 %d | >20 %d" % (
    sum(1 for v in gsize.values() if v == 1),
    sum(1 for v in gsize.values() if 2 <= v <= 5),
    sum(1 for v in gsize.values() if 6 <= v <= 10),
    sum(1 for v in gsize.values() if 11 <= v <= 20),
    sum(1 for v in gsize.values() if v > 20)))

TIGHT_MAX = 10


def genes_sharing(gs):
    s = set()
    for g, lst in g2groups.items():
        if set(lst) & set(gs):
            s.add(g)
    return s


# ---- DEF-C：每个基因取最特异（最小）家族作为唯一归属 ----
def most_specific(g):
    gs = g2groups[g]
    if not gs:
        return "SINGLETON_" + g
    best = min(gs, key=lambda x: (gsize[x], x))
    return "FAM:" + best


cluster_c = {g: most_specific(g) for g in genes}
cc = Counter(cluster_c.values())
print("\n[DEF-C 最特异家族] 簇数:", len(cc))
print("  单基因簇 %d | 2-5 %d | 6-10 %d | 11-20 %d | >20 %d" % (
    sum(1 for v in cc.values() if v == 1),
    sum(1 for v in cc.values() if 2 <= v <= 5),
    sum(1 for v in cc.values() if 6 <= v <= 10),
    sum(1 for v in cc.values() if 11 <= v <= 20),
    sum(1 for v in cc.values() if v > 20)))
print("  最大 8 个簇:", sorted(cc.items(), key=lambda x: -x[1])[:8])

# ---- DEF-B：只用 tight 家族（数据集内 ≤10 个基因）----
tight = {x for x, c in gsize.items() if c <= TIGHT_MAX}
g2tight = {g: [x for x in gs if x in tight] for g, gs in g2groups.items()}
print("\n[DEF-B tight 家族, 规模≤%d] 有 tight 家族的基因: %d / %d"
      % (TIGHT_MAX, sum(1 for v in g2tight.values() if v), len(genes)))

# ---- 模拟：随机取 20% 基因做测试，看三种定义下训练集要剔除多少 ----
import random
random.seed(42)
sh = genes[:]
random.shuffle(sh)
n_test = int(len(sh) * 0.20)
test_genes = set(sh[:n_test])
train_genes = set(sh[n_test:])
print("\n=== 模拟（测试基因 %d / 训练基因 %d）===" % (len(test_genes), len(train_genes)))


def removed_count(use_tight):
    tg_groups = set()
    for g in test_genes:
        tg_groups |= set(g2tight[g] if use_tight else g2groups[g])
    rem = set()
    for g in train_genes:
        lst = g2tight[g] if use_tight else g2groups[g]
        if set(lst) & tg_groups:
            rem.add(g)
    return len(rem), len(tg_groups)


for name, use_tight in [("DEF-A 全部家族", False), ("DEF-B tight", True)]:
    r, ng = removed_count(use_tight)
    print("  %-16s 测试基因涉及的家族 %d 个 ⇒ 需从训练集剔除 %d / %d 基因 (%.1f%%)，剩余 %d"
          % (name, ng, r, len(train_genes), 100 * r / len(train_genes),
             len(train_genes) - r))

out = {
    "_source": "HGNC gene_group (rest.genenames.org)，按 HGNC_ID 精确匹配",
    "_n_genes": len(genes),
    "_tight_max": TIGHT_MAX,
    "gene_to_groups": {g: g2groups[g] for g in genes},
    "group_size_in_dataset": dict(gsize),
    "cluster_C_most_specific": cluster_c,
    "tight_groups": sorted(tight),
}
json.dump(out, open(HERE + "/data/g3_gene_clusters.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print("\n已存 g3_gene_clusters.json")
