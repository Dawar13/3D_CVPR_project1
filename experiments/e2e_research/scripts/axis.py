import sys,os,glob,numpy as np
exec(open(os.path.join(os.path.dirname(sys.argv[0]),'drift.py')).read().split('d=sys.argv[1]')[0])
d=sys.argv[1]
for vid in ['42445441','47331068','42446038']:
    H=load(os.path.join(d,vid+'_hires.traj')); L=load(os.path.join(d,vid+'_lowres.traj')); T=np.load(os.path.join(d,vid+'_T.npy'),allow_pickle=False)
    k=sorted(H)[len(H)//2]; Ra=(np.linalg.inv(T)@L[k])[:3,:3]; Rf=H[k][:3,:3]
    D=Ra.T@Rf; w,v=np.linalg.eig(D); ax=np.real(v[:,np.argmin(abs(w-1))])
    print(vid,'R_arkit^T R_colmap =\n',np.round(D,3),' axis',np.round(ax,3))
