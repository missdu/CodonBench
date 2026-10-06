"""
WB | G3 真实同源/家族划分 —— 第 1 步：取 HGNC 基因家族注释

为什么用 HGNC：
  * genenames.org 是 HUGO 基因命名委员会的官方库，gene_group 字段是**人工审编的
    基因家族归属**（如 BBSome、Armadillo repeat containing），不是我们拍脑袋的定义
  * 服务器无外网，Ensembl REST / NCBI E-utilities 实测均不通；genenames.org 通
  * 关键是它**外生于模型**——不是从我们自己的嵌入算出来的，所以不构成循环论证

输入：g3_gene_list.csv（498 个基因，全部带 HGNC_ID）
输出：hgnc_gene_groups.json（基因 -> gene_group 列表 / gene_group_id 列表 / ensembl_gene_id）
"""
import json
import time
import urllib.request
import urllib.error
import csv
from collections import defaultdict

HERE = __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0]
GENE_CSV = HERE + "/data/g3_gene_list.csv"
OUT = HERE + "/data/hgnc_gene_groups.json"

rows = list(csv.DictReader(open(GENE_CSV, encoding="utf-8")))
print("基因行:", len(rows))

by_hgnc = {}
for r in rows:
    h = (r.get("HGNC_ID") or "").strip()
    if h:
        by_hgnc.setdefault(h, set()).add(r["gene_symbol"])
ids = sorted(by_hgnc)
print("唯一 HGNC ID:", len(ids))


def fetch(batch):
    url = "https://rest.genenames.org/fetch/hgnc_id/" + ",".join(batch)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=40) as f:
                return json.load(f)
        except Exception as e:
            print("   重试 %d: %s" % (attempt + 1, str(e)[:80]))
            time.sleep(2 + 3 * attempt)
    return None


out = {}
BS = 20
for i in range(0, len(ids), BS):
    batch = ids[i:i + BS]
    d = fetch(batch)
    if d is None:
        print("  ❌ 批次 %d 失败，跳过 %d 个" % (i, len(batch)))
        continue
    for doc in d.get("response", {}).get("docs", []):
        out[doc.get("hgnc_id")] = {
            "symbol": doc.get("symbol"),
            "gene_group": doc.get("gene_group", []),
            "gene_group_id": doc.get("gene_group_id", []),
            "ensembl_gene_id": doc.get("ensembl_gene_id"),
            "locus_type": doc.get("locus_type"),
            "uniprot_ids": doc.get("uniprot_ids", []),
        }
    print("  已取 %d / %d" % (min(i + BS, len(ids)), len(ids)))
    time.sleep(0.3)

print("\n成功取回:", len(out), "/", len(ids))
miss = [h for h in ids if h not in out]
if miss:
    print("未取到:", miss[:10], "...")

# 统计
withg = sum(1 for v in out.values() if v["gene_group"])
print("有家族注释的基因:", withg, "/", len(out))

gcount = defaultdict(int)
for v in out.values():
    for g in v["gene_group"]:
        gcount[g] += 1
print("\n家族总数:", len(gcount))
print("最大的 15 个家族（按本数据集中基因数）:")
for g, c in sorted(gcount.items(), key=lambda x: -x[1])[:15]:
    print("   %-55s %d" % (g, c))

json.dump({"by_hgnc_id": out,
           "gene_to_hgnc": {s: h for h, ss in by_hgnc.items() for s in ss}},
          open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\n已存:", OUT)
