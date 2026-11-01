import sys,os,glob,numpy as np
exec(open(os.path.join(os.path.dirname(sys.argv[0]),'drift.py')).read().split('d=sys.argv[1]')[0])
Rz=np.diag([-1,-1,1.0])
def ang(R): return np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
d=sys.argv[1]
for f in sorted(glob.glob(os.path.join(d,'*_hires.traj'))):
    vid=os.path.basename(f).split('_')[0]
    H=load(f); L=load(os.path.join(d,vid+'_lowres.traj'))
    ks=sorted(H)
    # check flip about optical axis: compare relative rotations (flip cancels if constant)
    out={}
    for gap in [10,30]:  # frames at ~10 Hz -> ~1 s, ~3 s
        re=[];te=[];tn=[]
        for i in range(0,len(ks)-gap,3):
            a,b=ks[i],ks[i+gap]
            Rh=np.linalg.inv(H[a])@H[b]; Rl=np.linalg.inv(L[a])@L[b]
            # flip conjugation: hires cam frame may be rotated 180 deg about z vs lowres
            cands=[Rh, np.diag([-1,-1,1,1])@Rh@np.diag([-1,-1,1,1])]
            e=[ang(c[:3,:3].T@Rl[:3,:3]) for c in cands]; k=int(np.argmin(e)); Rhc=cands[k]
            re.append(e[k]); te.append(np.linalg.norm(Rhc[:3,3]-Rl[:3,3])); tn.append(np.linalg.norm(Rhc[:3,3]))
        re=np.array(re);te=np.array(te);tn=np.array(tn)
        out[gap]=f'gap~{gap/10:.0f}s: rel-rot err med {np.median(re):.2f} p90 {np.percentile(re,90):.2f} deg; rel-trans err med {np.median(te)*100:.1f} p90 {np.percentile(te,90)*100:.1f} cm (baseline med {np.median(tn)*100:.0f} cm)'
    print(vid,'|',out[10],'|',out[30])
