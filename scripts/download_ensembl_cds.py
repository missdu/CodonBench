"""
Download human protein-coding CDS sequences from Ensembl.

Uses Ensembl BioMart REST API to fetch all human protein-coding gene CDS sequences.
This provides real biological CDS data (not synthetic) for pretraining ablation experiments.

Usage:
    python download_ensembl_cds.py [--output OUTPUT] [--max_seqs MAX_SEQS]

Output: JSON file with {"sequences": [...], "source": "Ensembl", "species": "homo_sapiens", ...}
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET


ENSEMBL_BIOMART_URL = "http://www.ensembl.org/biomart/martservice"


def build_mart_xml(max_seqs=None):
    dataset = "hsapiens_gene_ensembl"
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="FASTA" header="0" uniqueRows="1" count="" datasetConfigVersion="0.6">
    <Dataset name="{dataset}" interface="default">
        <Filter name="biotype" value="protein_coding"/>
        <Attribute name="ensembl_transcript_id"/>
        <Attribute name="coding"/>
    </Dataset>
</Query>"""
    return xml


def download_via_biomart(output_path, max_seqs=None):
    print("Downloading CDS sequences from Ensembl BioMart...")
    print("This may take 10-30 minutes depending on network speed.")

    xml_query = build_mart_xml(max_seqs)
    params = urllib.parse.urlencode({"query": xml_query})

    url = f"{ENSEMBL_BIOMART_URL}?{params}"

    tmp_path = output_path + ".tmp"
    try:
        urllib.request.urlretrieve(url, tmp_path)
    except Exception as e:
        print(f"BioMart download failed: {e}")
        print("Trying alternative: Ensembl REST API...")
        return download_via_rest_api(output_path, max_seqs)

    sequences = []
    current_id = None
    current_seq = []

    with open(tmp_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id and current_seq:
                    seq = "".join(current_seq).upper().replace(" ", "").replace("\n", "")
                    if len(seq) >= 30 and all(c in "ATGC" for c in seq):
                        sequences.append(seq)
                        if max_seqs and len(sequences) >= max_seqs:
                            break
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        if current_id and current_seq:
            seq = "".join(current_seq).upper().replace(" ", "").replace("\n", "")
            if len(seq) >= 30 and all(c in "ATGC" for c in seq):
                sequences.append(seq)

    os.remove(tmp_path)

    print(f"Downloaded {len(sequences)} CDS sequences from Ensembl BioMart")
    return sequences


def download_via_rest_api(output_path, max_seqs=None):
    """Alternative: use Ensembl REST API to fetch CDS by gene IDs."""
    import urllib.request

    base_url = "https://rest.ensembl.org"

    print("Fetching human gene list from Ensembl...")
    gene_url = f"{base_url}/overlap/species/homo_sapiens?feature=gene;biotype=protein_coding;content-type=application/json"
    
    try:
        req = urllib.request.Request(gene_url)
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=60) as response:
            genes = json.loads(response.read().decode())
    except Exception as e:
        print(f"Failed to fetch gene list: {e}")
        return []

    print(f"Found {len(genes)} protein-coding genes")

    sequences = []
    seen_transcripts = set()

    for i, gene in enumerate(genes):
        if max_seqs and len(sequences) >= max_seqs:
            break

        gene_id = gene.get("id")
        if not gene_id:
            continue

        try:
            seq_url = f"{base_url}/sequence/id/{gene_id}?type=cds;content-type=application/json"
            req = urllib.request.Request(seq_url)
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=30) as response:
                seq_data = json.loads(response.read().decode())

            if isinstance(seq_data, dict):
                seq_data = [seq_data]

            for item in seq_data:
                seq = item.get("seq", "").upper().replace(" ", "")
                tid = item.get("id", "")
                if tid in seen_transcripts:
                    continue
                seen_transcripts.add(tid)

                if len(seq) >= 30 and all(c in "ATGC" for c in seq):
                    sequences.append(seq)

        except Exception as e:
            if i % 1000 == 0:
                print(f"  Gene {i}/{len(genes)}: {len(sequences)} CDS so far")
            continue

        if i % 1000 == 0:
            print(f"  Gene {i}/{len(genes)}: {len(sequences)} CDS so far")

        if i % 5 == 0:
            time.sleep(0.1)

    print(f"Downloaded {len(sequences)} CDS sequences from Ensembl REST API")
    return sequences


def download_via_ensembl_archive(output_path, max_seqs=None):
    """Download from Ensembl FTP - most reliable for bulk data."""
    import gzip

    ftp_url = "https://ftp.ensembl.org/pub/release-112/fasta/homo_sapiens/cds/Homo_sapiens.GRCh38.cds.all.fa.gz"

    print(f"Downloading from Ensembl FTP: {ftp_url}")
    tmp_gz = output_path + ".fa.gz"

    try:
        urllib.request.urlretrieve(ftp_url, tmp_gz)
    except Exception as e:
        print(f"FTP download failed: {e}")
        return None

    print("Parsing FASTA file...")
    sequences = []

    with gzip.open(tmp_gz, "rt") as f:
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    seq = "".join(current_seq).upper().replace(" ", "")
                    if len(seq) >= 30 and all(c in "ATGC" for c in seq):
                        sequences.append(seq)
                        if max_seqs and len(sequences) >= max_seqs:
                            break
                current_seq = []
            else:
                current_seq.append(line)

        if current_seq:
            seq = "".join(current_seq).upper().replace(" ", "")
            if len(seq) >= 30 and all(c in "ATGC" for c in seq):
                sequences.append(seq)

    os.remove(tmp_gz)
    print(f"Parsed {len(sequences)} CDS sequences from Ensembl FTP")
    return sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.expanduser(
        "./data/ensembl_cds/ensembl_human_cds.json"
    ))
    parser.add_argument("--max_seqs", type=int, default=None)
    parser.add_argument("--method", choices=["ftp", "biomart", "rest"], default="ftp")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.method == "ftp":
        sequences = download_via_ensembl_archive(args.output, args.max_seqs)
    elif args.method == "biomart":
        sequences = download_via_biomart(args.output, args.max_seqs)
    else:
        sequences = download_via_rest_api(args.output, args.max_seqs)

    if sequences is None or len(sequences) == 0:
        print("All download methods failed. Trying FTP as fallback...")
        sequences = download_via_ensembl_archive(args.output, args.max_seqs)

    if sequences is None or len(sequences) == 0:
        print("ERROR: Could not download any CDS sequences.")
        sys.exit(1)

    data = {
        "sequences": sequences,
        "source": "Ensembl",
        "species": "homo_sapiens",
        "release": "112",
        "assembly": "GRCh38",
        "num_sequences": len(sequences),
        "total_bases": sum(len(s) for s in sequences),
        "avg_length": sum(len(s) for s in sequences) / len(sequences),
    }

    with open(args.output, "w") as f:
        json.dump(data, f)

    print(f"\nSaved {len(sequences)} CDS sequences to {args.output}")
    print(f"Total bases: {data['total_bases']:,}")
    print(f"Average length: {data['avg_length']:.0f} bp")


if __name__ == "__main__":
    main()