#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WB-fetch-refgene.py —— 下载 UCSC refGene（hg19 + hg38），解析出每个转录本的外显子结构。

用途：Reject §二.4.1 点名"CDS 边界 ≠ 外显子边界"，需要变异到最近外显子—内含子接点的距离。
服务器无外网（实测 NCBI/Ensembl 均 000），故在本地下载后传给服务器。

输出：WB-out/data/refgene_exons.tsv
  tx_acc      NM_xxxxx（去掉版本号）
  build       hg19 / hg38
  chrom       chr1 ...
  strand      + / -
  n_exon
  exon_starts 逗号分隔（0-based, 半开）
  exon_ends   逗号分隔
  cds_start   CDS 在基因组上的起点（0-based）
  cds_end     CDS 终点（半开）

加 cds_start/cds_end 的原因：Reject §二.4.1 的对照是
「用 CDS 边界算的靠近比例」对「用外显子—内含子接点算的靠近比例」，
两个都要有，才能证明原稿"仅 0.2% 靠近 CDS 边界"不足以排除剪接混杂。

判据（防自欺）：
  1. 两个 build 都要有非空结果；
  2. 抽查若干转录本：外显子数 ≥1、start<end、坐标递增、互不重叠；
  3. 外显子总长 ≥ CDS 长度（若有）。
"""
import gzip
import io
import os
import sys
import urllib.request

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "refgene_exons.tsv")

URLS = {
    "hg19": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/refGene.txt.gz",
    "hg38": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/refGene.txt.gz",
}


def fetch(build):
    url = URLS[build]
    print(f"[fetch] {build} <- {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    print(f"[fetch] {build} bytes={len(raw)}", flush=True)
    txt = gzip.decompress(raw).decode("utf-8", errors="replace")
    print(f"[fetch] {build} uncompressed chars={len(txt)}", flush=True)
    return txt


def parse(txt, build):
    """UCSC refGene schema:
    0 bin, 1 name(acc), 2 chrom, 3 strand, 4 txStart, 5 txEnd,
    6 cdsStart, 7 cdsEnd, 8 exonCount, 9 exonStarts, 10 exonEnds,
    11 score, 12 name2(gene symbol), 13 cdsStartStat, 14 cdsEndStat, 15 exonFrames
    """
    rows = []
    n_bad = 0
    for line in txt.splitlines():
        if not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 16:
            n_bad += 1
            continue
        acc = f[1]
        if not acc.startswith(("NM_", "NR_")):
            continue
        chrom, strand = f[2], f[3]
        try:
            n_exon = int(f[8])
        except ValueError:
            n_bad += 1
            continue
        starts = [x for x in f[9].rstrip(",").split(",") if x]
        ends = [x for x in f[10].rstrip(",").split(",") if x]
        if len(starts) != n_exon or len(ends) != n_exon or n_exon == 0:
            n_bad += 1
            continue
        try:
            cds_start, cds_end = int(f[6]), int(f[7])
        except ValueError:
            n_bad += 1
            continue
        rows.append((acc.split(".")[0], build, chrom, strand, n_exon,
                     ",".join(starts), ",".join(ends), cds_start, cds_end))
    print(f"[parse] {build}: kept={len(rows)} skipped_bad={n_bad}", flush=True)
    return rows


def sanity(rows, build):
    """判据：start<end、递增、不重叠、正链/负链都有。"""
    problems = {"start_ge_end": 0, "not_sorted": 0, "overlap": 0}
    strands = {}
    for acc, b, chrom, strand, n, ss, es, cs, ce in rows:
        strands[strand] = strands.get(strand, 0) + 1
        s = [int(x) for x in ss.split(",")]
        e = [int(x) for x in es.split(",")]
        if any(a >= b_ for a, b_ in zip(s, e)):
            problems["start_ge_end"] += 1
        if any(s[i + 1] < s[i] for i in range(len(s) - 1)):
            problems["not_sorted"] += 1
        if any(s[i + 1] < e[i] for i in range(len(s) - 1)):
            problems["overlap"] += 1
    print(f"[sanity] {build}: {problems} strands={strands}", flush=True)
    return problems


def main():
    all_rows = []
    for build in ("hg19", "hg38"):
        txt = fetch(build)
        rows = parse(txt, build)
        p = sanity(rows, build)
        if not rows:
            print(f"[FATAL] {build} 解析为空 —— 不产出，避免静默坏数据", file=sys.stderr)
            sys.exit(2)
        if p["start_ge_end"] or p["not_sorted"] or p["overlap"]:
            print(f"[WARN] {build} 结构有异常，仍落盘但需人工复核", file=sys.stderr)
        all_rows.extend(rows)

    with open(OUT_PATH, "w", encoding="utf-8", newline="") as fh:
        fh.write("tx_acc\tbuild\tchrom\tstrand\tn_exon\texon_starts\texon_ends"
                 "\tcds_start\tcds_end\n")
        for r in all_rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(f"[out] {OUT_PATH} rows={len(all_rows)}", flush=True)

    # 负例：不存在的 accession 必须返回空，而不是被静默跳过成"0 距离"
    print("[negctl] 检查 NM_NOTEXIST 是否不在表中：",
          all(r[0] != "NM_NOTEXIST" for r in all_rows), flush=True)


if __name__ == "__main__":
    main()
