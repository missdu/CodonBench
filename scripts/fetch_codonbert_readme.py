#!/usr/bin/env python3
"""Fetch CodonBERT HF dataset README to find fungal expression data source."""
import urllib.request
import os
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

try:
    url = "https://huggingface.co/datasets/lhallee/CodonBERT/raw/main/README.md"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=30)
    text = resp.read().decode("utf-8")
    print(text[:5000])
except Exception as e:
    print(f"Error: {e}")
    try:
        url2 = "https://raw.githubusercontent.com/microsoft/CodonBERT/main/README.md"
        req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
        resp2 = urllib.request.urlopen(req2, timeout=30)
        text2 = resp2.read().decode("utf-8")
        print(text2[:5000])
    except Exception as e2:
        print(f"GitHub also failed: {e2}")