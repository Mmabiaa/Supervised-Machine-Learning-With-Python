from pathlib import Path
import zipfile
from xml.etree import ElementTree as ET
p = Path('Supervised_ML_with Python.docx')
with zipfile.ZipFile(p) as z:
    xml = z.read('word/document.xml')
root = ET.fromstring(xml)
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
texts = []
for paragraph in root.findall('.//w:p', ns):
    parts = []
    for t in paragraph.findall('.//w:t', ns):
        parts.append(t.text or '')
    txt = ''.join(parts)
    if txt.strip():
        texts.append(txt)
for i, t in enumerate(texts, 1):
    print(f'{i}: {t}')
