"""Combine the reviewed local solutions without changing downloaded originals."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent
PARTS = [
    '作业1_VAE_解析.md',
    '作业2_AR与Transformer_解析.md',
    '作业3_GAN_解析.md',
    '作业4_扩散模型_解析.md',
    '作业5_语言模型微调_解析.md',
    '期中2024_逐题解析.md',
    '期中2025_逐题解析.md',
    'GitHub资料_解析与勘误.md',
]
intro = '''# 生成模型基础全题解析

覆盖五次作业与 2024、2025 两年期中卷，按原题共 58 个作答单元，另含 2025 PDF 附解中的 3 道额外题。各题先列完整原文、条件与 PDF 页码，再给出中文详细解析、中间推导、手算例子及易错点。共同题设保留在小问之前。最后附 GitHub 同课程参考资料的来源、逐题对应和勘误。这是学习解析，不是课程官方标准答案。

分卷入口与覆盖表见 [解析目录](README.md)，数值校验见 [核验记录](核验记录.md)。

'''
pieces = [intro]
for name in PARTS:
    text = (ROOT / name).read_text(encoding='utf-8')
    # Shift all Markdown headings one level below the collection title.
    text = re.sub(r'^(#{1,5}) ', r'#\1 ', text, flags=re.M)
    pieces.append(text.rstrip() + '\n')
combined = ROOT / '全题解析.md'
# A published checkout uses this file as a short navigation index. Rebuilding
# the two volumes must not replace that index with an oversized online page.
if not combined.exists() or '## 合订版分卷阅读' not in combined.read_text(encoding='utf-8'):
    combined.write_text('\n\n'.join(pieces), encoding='utf-8')

for filename, title, selected in [
    ('全题解析_作业卷.md', '生成模型基础全题解析 作业卷', PARTS[:5]),
    ('全题解析_期中卷.md', '生成模型基础全题解析 期中卷', PARTS[5:7]),
]:
    sections = [f'# {title}\n\n每题包含 PDF 原文、条件、页码、中文详细解析及进一步推导。'
                '本卷保留各分题文件的完整内容。\n\n'
                '另见 [作业卷](全题解析_作业卷.md)、[期中卷](全题解析_期中卷.md)、'
                '[GitHub资料与勘误](GitHub资料_解析与勘误.md)及[分题目录](README.md)。\n']
    for name in selected:
        text = (ROOT / name).read_text(encoding='utf-8')
        text = re.sub(r'^(#{1,5}) ', r'#\1 ', text, flags=re.M)
        sections.append(text.rstrip() + '\n')
    (ROOT / filename).write_text('\n\n'.join(sections), encoding='utf-8')

source_root = ROOT.parent / 'GitHub资料'
manifest_path = source_root / 'downloads.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
snapshots = {
    'joerunrun/generative_model': '83f1dcda1a1a1b7c8cd1867c80e58b0e32632468',
    'TidalHarley/PKU-Genai-notes': '47df72c040ac19ef87bb94cfacf845a0bbc408ce',
}
for row in manifest:
    p = source_root / row['local'].replace('\\', '/')
    data = p.read_bytes()
    assert hashlib.sha256(data).hexdigest() == row['sha256']
    row['git_blob_sha1'] = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    row['repository_snapshot_sha'] = snapshots[row['repo']]
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('Built 全题解析.md:', (ROOT / '全题解析.md').stat().st_size, 'bytes')
print('Recorded exact repository snapshots and Git blob hashes for', len(manifest), 'files')
