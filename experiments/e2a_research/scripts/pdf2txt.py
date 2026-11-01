import sys, fitz
src, dst = sys.argv[1], sys.argv[2]
doc = fitz.open(src)
with open(dst, 'w', encoding='utf-8') as f:
    for i, p in enumerate(doc):
        f.write(f"\n=== PAGE {i+1} ===\n")
        f.write(p.get_text())
print(len(doc))
