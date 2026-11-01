import sys,os,glob,numpy as np
def rod(r):
    th=np.linalg.norm(r)
    if th<1e-12: return np.eye(3)
    k=r/th; K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)+np.sin(th)*K+(1-np.cos(th))*K@K
for f in sorted(glob.glob(os.path.join(sys.argv[1],'*.traj'))):
    a=np.loadtxt(f); ts=a[:,0]; dt=np.diff(ts)
    C=[];X=[];Y=[];Z=[]
    for row in a:
        R=rod(row[1:4]); t=row[4:7]
        Rc=R.T; C.append(-Rc@t); X.append(Rc[:,0]); Y.append(Rc[:,1]); Z.append(Rc[:,2])
    C=np.array(C);X=np.array(X);Y=np.array(Y);Z=np.array(Z)
    print(os.path.basename(f),'n',len(a),'dur %.1fs'%(ts[-1]-ts[0]),'dt median %.4f min %.4f max %.4f'%(np.median(dt),dt.min(),dt.max()))
    print('  first centre',np.round(C[0],3),'|C0|=%.3f'%np.linalg.norm(C[0]),' extent min',np.round(C.min(0),2),'max',np.round(C.max(0),2))
    print('  mean |world comp| of cam x-axis',np.round(np.abs(X).mean(0),2),' cam y-axis',np.round(np.abs(Y).mean(0),2),' optical z-axis',np.round(np.abs(Z).mean(0),2))
    print('  mean signed cam y-axis',np.round(Y.mean(0),2),' cam x-axis',np.round(X.mean(0),2))
    print('  first R_c2w\n',np.round(rod(a[0,1:4]).T,3))
