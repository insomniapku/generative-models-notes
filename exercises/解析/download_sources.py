"""Download only verified public course resources; never execute downloaded code."""
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import quote
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1] / 'GitHub资料'
SOURCES = [
    ('joerunrun/generative_model', 'main', 'README.md'),
    ('joerunrun/generative_model', 'main', 'midtest.pdf'),
    ('joerunrun/generative_model', 'main', 'midtest.tex'),
    ('TidalHarley/PKU-Genai-notes', 'main', 'README.md'),
    ('TidalHarley/PKU-Genai-notes', 'main', '生成模型基础 Basic of Generative Model (25 Fall).pdf'),
]

def main():
    ROOT.mkdir(exist_ok=True)
    records = []
    for repo, ref, path in SOURCES:
        target = ROOT / repo.replace('/', '__') / path
        target.parent.mkdir(parents=True, exist_ok=True)
        url = 'https://raw.githubusercontent.com/' + repo + '/' + ref + '/' + quote(path)
        data = urlopen(Request(url, headers={'User-Agent': 'CourseStudyArchive/1.0'}), timeout=50).read()
        if path.endswith('.pdf') and not data.startswith(b'%PDF'):
            raise ValueError('Invalid PDF: ' + url)
        target.write_bytes(data)
        record = dict(repo=repo, ref=ref, path=path, url=url,
                      local=str(target.relative_to(ROOT)), bytes=len(data),
                      sha256=hashlib.sha256(data).hexdigest(), download_date='2026-11-17')
        records.append(record)
        (ROOT / 'downloads.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
        print(str(target.relative_to(ROOT)), len(data), 'bytes', flush=True)

if __name__ == '__main__':
    main()
