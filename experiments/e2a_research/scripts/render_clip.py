import sys, fitz
src, page, dst, zoom = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4])
x0,y0,x1,y1 = [float(v) for v in sys.argv[5:9]]
doc = fitz.open(src)
p = doc[page]
r = p.rect
print("rect", r)
clip = fitz.Rect(r.x0 + x0*r.width, r.y0 + y0*r.height, r.x0 + x1*r.width, r.y0 + y1*r.height)
pix = p.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
pix.save(dst); print(pix.width, pix.height)
