import urllib.request

url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=11267658&rettype=xml'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
resp = urllib.request.urlopen(req, timeout=30)
text = resp.read().decode('utf-8', errors='ignore')

keywords = ['Fungal', 'fungal', 'Gish', 'yeast', 'Saccharomyces', '7089',
            'E. coli', 'E.coli', 'mRFP', 'mRNA stability', 'expression data',
            'regression', 'protein expression']
for kw in keywords:
    idx = text.find(kw)
    if idx >= 0:
        print(f'Found "{kw}" at {idx}: ...{text[max(0,idx-200):idx+200]}...')
        print('---')