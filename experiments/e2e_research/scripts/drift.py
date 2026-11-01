import sys,os,glob,numpy as np
def rod(r):
    th=np.linalg.norm(r)
    if th<1e-12: return np.eye(3)
    k=r/th; K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)+np.sin(th)*K+(1-np.cos(th))*K@K
def load(f):
    out={}
    for row in np.loadtxt(f):
        E=np.eye(4); E[:3,:3]=rod(row[1:4]); E[:3,3]=row[4:7]
        out[round(row[0],3)]=np.linalg.inv(E)  # cam->world
    return out
d=sys.argv[1]
for f in sorted(glob.glob(os.path.join(d,'*_hires.traj'))):
    vid=os.path.basename(f).split('_')[0]
    H=load(f); L=load(os.path.join(d,vid+'_lowres.traj'))
    T=np.load(os.path.join(d,vid+'_T.npy'),allow_pickle=False)
    sc=np.cbrt(np.linalg.det(T[:3,:3]))
    common=sorted(set(H)&set(L))
    # also nearest within 5 ms
    hk=np.array(sorted(H)); lk=np.array(sorted(L))
    near=[(h,lk[np.argmin(abs(lk-h))]) for h in hk]
    near=[(h,l) for h,l in near if abs(h-l)<0.005]
    Ti=np.linalg.inv(T)
    rot=[];pos=[];tt=[]
    for h,l in near:
        Pf=H[h]; Pa=Ti@L[l]
        Ra=Pa[:3,:3]/np.cbrt(np.linalg.det(Pa[:3,:3]))
        dR=Ra.T@Pf[:3,:3]; ang=np.degrees(np.arccos(np.clip((np.trace(dR)-1)/2,-1,1)))
        rot.append(ang); pos.append(np.linalg.norm(Pa[:3,3]-Pf[:3,3])); tt.append(h)
    rot=np.array(rot);pos=np.array(pos);tt=np.array(tt)-tt[0]
    C=np.array([H[h][:3,3] for h in hk])
    print(f'{vid}: hires n={len(H)} lowres n={len(L)} exact-ts matches={len(common)} matches<5ms={len(near)}; T scale={sc:.4f}')
    print(f'   hires cam-centre extent {np.round(C.min(0),2)} .. {np.round(C.max(0),2)}; hires dt median {np.median(np.diff(hk)):.3f}')
    q=len(rot)//4
    print(f'   ARKit(lowres, mapped by T^-1) vs COLMAP(hires): rot err median {np.median(rot):.2f} deg, p90 {np.percentile(rot,90):.2f}, max {rot.max():.2f}; pos err median {np.median(pos)*100:.1f} cm, p90 {np.percentile(pos,90)*100:.1f} cm, max {pos.max()*100:.1f} cm')
    print(f'   by quarter of video: rot med {[round(float(np.median(rot[i*q:(i+1)*q])),2) for i in range(4)]} deg; pos med {[round(float(np.median(pos[i*q:(i+1)*q])*100),1) for i in range(4)]} cm')
