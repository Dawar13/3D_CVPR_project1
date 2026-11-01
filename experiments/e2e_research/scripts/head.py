import sys,urllib.request,concurrent.futures as cf,collections
lines=[l.split() for l in open(sys.argv[1]) if l.strip()]
def head(x):
    vid,name,url=x
    for k in range(3):
        try:
            r=urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=60)
            return vid,name,r.status,int(r.headers.get('Content-Length',0))
        except urllib.error.HTTPError as e: return vid,name,e.code,0
        except Exception as e: err=str(e)
    return vid,name,'ERR',0
with cf.ThreadPoolExecutor(6) as ex: res=list(ex.map(head,lines))
with open(sys.argv[2],'w') as f:
    for r in res: f.write(' '.join(map(str,r))+'\n')
s=collections.defaultdict(int);n=collections.Counter();ok=collections.Counter()
for vid,name,st,sz in res:
    key='sf_refined_transform.npy' if name.endswith('refined_transform.npy') else name
    n[key]+=1; s[key]+=sz; ok[key]+= (st==200)
for k in sorted(s): print(f'{k:32s} ok={ok[k]}/{n[k]} total={s[k]/1e9:.2f} GB mean={s[k]/n[k]/1e6:.2f} MB')
