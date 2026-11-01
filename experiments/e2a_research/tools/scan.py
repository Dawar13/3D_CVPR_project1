import json, sys, re
kw = re.compile(r"(nois|corrupt|perturb|inaccurate|erroneous|wrong|unreliab|robust|outlier|drift|prior|condition|hint|auxiliary|calibrat|intrinsic|uncertain|trust|reliab|faulty|imperfect|misspecif|adversar)", re.I)
strong = re.compile(r"(noisy prior|noisy pose|noisy camera|noisy depth|noisy intrinsic|prior noise|corrupted prior|inaccurate prior|imperfect prior|unreliable prior|wrong prior|erroneous|prior reliab|prior robust|robust to .*prior|noisy geometric|noisy input|pose noise|perturbed pose|perturbed prior|drift|prior dilution|any-prior|prior inject|prior-aware|camera prior|depth prior|pose prior|geometric prior|auxiliary input|optional input)", re.I)
for fn in sys.argv[1:]:
    d = json.load(open(fn, encoding="utf-8"))
    rows = d["data"]
    print("FILE", fn, "n=", len(rows))
    hits = []
    for r in rows:
        p = r["citingPaper"]
        t = p.get("title") or ""
        a = p.get("abstract") or ""
        aid = (p.get("externalIds") or {}).get("ArXiv")
        s = strong.findall(t + " " + a)
        if s:
            hits.append((len(set(x.lower() for x in s)), aid, p.get("year"), t, sorted(set(x.lower() for x in s))))
    hits.sort(key=lambda x: -x[0])
    print("strong-keyword hits:", len(hits))
    for h in hits:
        print(f"{h[1]} | {h[2]} | {h[3]} | {h[4]}")
