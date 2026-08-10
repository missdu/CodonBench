import urllib.request

url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=11267658&rettype=xml'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
resp = urllib.request.urlopen(req, timeout=30)
text = resp.read().decode('utf-8', errors='ignore')

keywords = ['180', '10 million', '10M', 'pretraining', 'pre-training', 'pretrain', 'codon sequence']
for kw in keywords:
    idx = 0
    while True:
        idx = text.lower().find(kw.lower(), idx)
        if idx < 0:
            break
        context = text[max(0,idx-200):idx+200]
        if any(c.isalpha() for c in context):
            print(f'Found "{kw}" at {idx}: ...{context}...')
            print('---')
        idx += len(kw)
        if idx > len(text):
            break