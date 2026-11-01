import sys, fitz
src, page, dst, zoom = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4])
doc = fitz.open(src)
print("pages", len(doc))
pix = doc[page].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
pix.save(dst)
