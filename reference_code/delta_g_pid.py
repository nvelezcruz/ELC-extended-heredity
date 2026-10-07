import numpy as np
from scipy.special import ndtri


def _as2d(x):
    x=np.asarray(x,float)
    if x.ndim==1: x=x[:,None]
    return x

def rank_gauss(x):
    x=_as2d(x); n=x.shape[0]; z=np.zeros_like(x,dtype=float)
    for j in range(x.shape[1]):
        v=x[:,j]; order=np.argsort(v, kind='mergesort'); sv=v[order]
        bounds=np.r_[0, np.flatnonzero(sv[1:]!=sv[:-1])+1, n]
        ranks=np.empty(n,float)
        for a,b in zip(bounds[:-1],bounds[1:]): ranks[order[a:b]]=0.5*(a+b-1)+1
        u=np.clip((ranks-0.5)/n,1e-6,1-1e-6); z[:,j]=ndtri(u)
    return z

def cov(x):
    x=_as2d(x); x=x-x.mean(0,keepdims=True)
    return (x.T@x)/(x.shape[0]-1)

def sym(A): return (A+A.T)/2

def eig_clip(A, eps=1e-9):
    A=sym(A); w,V=np.linalg.eigh(A); w=np.maximum(w,eps); return V@np.diag(w)@V.T

def sqrtm(A, eps=1e-9):
    A=sym(A); w,V=np.linalg.eigh(A); w=np.maximum(w,eps); return V@np.diag(np.sqrt(w))@V.T

def invsqrtm(A, eps=1e-9):
    A=sym(A); w,V=np.linalg.eigh(A); w=np.maximum(w,eps); return V@np.diag(1/np.sqrt(w))@V.T

def logdet_spd(A, eps=1e-9):
    A=sym(A); w=np.linalg.eigvalsh(A); w=np.maximum(w,eps); return float(np.sum(np.log(w)))

def gaussian_mi(A,B):
    A=_as2d(A); B=_as2d(B); AB=np.hstack([A,B])
    return 0.5*(logdet_spd(cov(A))+logdet_spd(cov(B))-logdet_spd(cov(AB)))/np.log(2)

def bias_entropy_bits(d,n):
    if d<=0: return 0.0
    return 0.5*sum(np.log(max(1e-12,1-k/n)) for k in range(1,d+1))/np.log(2)

def bias_mi_bits(dA,dB,n):
    return bias_entropy_bits(dA,n)+bias_entropy_bits(dB,n)-bias_entropy_bits(dA+dB,n)

def project_spectral(U, max_s=1-1e-7):
    # project onto spectral norm ball ||U||_2 <= max_s
    if U.size==0: return U
    A=np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
    u,s,vt=np.linalg.svd(A, full_matrices=False)
    s=np.minimum(s,max_s)
    return u@np.diag(s)@vt

def preprocess(M,X,Y, rank_transform=True, ridge=1e-7):
    M=_as2d(M); X=_as2d(X); Y=_as2d(Y)
    if rank_transform:
        M=rank_gauss(M); X=rank_gauss(X); Y=rank_gauss(Y)
    M=M-M.mean(0,keepdims=True); X=X-X.mean(0,keepdims=True); Y=Y-Y.mean(0,keepdims=True)
    n=M.shape[0]; dM=M.shape[1]; dX=X.shape[1]; dY=Y.shape[1]
    # whiten M
    WM=invsqrtm(cov(M)+ridge*np.eye(dM))
    Mw=M@WM.T
    # regress channels H and residual covariances
    # H = Cov(X,M) Cov(M)^-1 = Cov(X,Mw) because Mw covariance approx I.
    HX0=cov(np.hstack([X,Mw]))[:dX,dX:]
    HY0=cov(np.hstack([Y,Mw]))[:dY,dY:]
    NX=X-Mw@HX0.T
    NY=Y-Mw@HY0.T
    WX=invsqrtm(cov(NX)+ridge*np.eye(dX))
    WY=invsqrtm(cov(NY)+ridge*np.eye(dY))
    Xw=X@WX.T; Yw=Y@WY.T
    HX=WX@HX0
    HY=WY@HY0
    return Mw,Xw,Yw,HX,HY

def _deficiency_one(HX, HY, max_iter=2500, lr=0.03, tol=1e-8, verbose=False):
    # Computes \hat{delta}_G(M:Y\X) in nats from Def 6.
    dX,dM=HX.shape; dY=HY.shape[0]
    Ix=np.eye(dX); Iy=np.eye(dY)
    A=Iy + HY@HY.T
    B=Ix + HX@HX.T
    A=sym(A)+1e-8*Iy; B=sym(B)+1e-8*Ix
    A_inv_sqrt=invsqrtm(A); A_sqrt=sqrtm(A); B_inv_sqrt=invsqrtm(B); B_sqrt=sqrtm(B)
    G=B_inv_sqrt@HX              # dX x dM
    D=A_inv_sqrt@HY              # dY x dM
    # objective || U G - D ||_F^2 / 2? Eq18 lacks 1/2 but minimizer same. Use F=0.5|| ||^2.
    # Initialize unconstrained solution projected.
    try:
        U=D@G.T@np.linalg.pinv(G@G.T + 1e-8*np.eye(dX))
    except Exception:
        U=np.zeros((dY,dX))
    U=project_spectral(U)
    bestU=U.copy(); R=U@G-D; best=0.5*np.sum(R*R)
    prev=best
    for it in range(max_iter):
        R=U@G-D
        grad=R@G.T
        # simple backtracking projected gradient
        step=lr
        improved=False
        for _ in range(20):
            Un=project_spectral(U-step*grad)
            Rn=Un@G-D; val=0.5*np.sum(Rn*Rn)
            if val <= prev + 1e-12:
                improved=True; break
            step*=0.5
        if not improved:
            Un=project_spectral(U-lr*0.1*grad); Rn=Un@G-D; val=0.5*np.sum(Rn*Rn)
        U=Un; prev=val
        if val<best:
            best=val; bestU=U.copy()
        if verbose and it%200==0: print('def',it,val,best,np.linalg.norm(grad))
        if np.linalg.norm(grad) < tol:
            break
    U=bestU
    T=A_sqrt@U@B_inv_sqrt
    Sigma_T=A - T@B@T.T
    Sigma_T=eig_clip(Sigma_T,1e-8)
    V=Sigma_T + T@T.T
    V=eig_clip(V,1e-8)
    diff=T@HX - HY
    val=0.5*(np.trace(diff@diff.T@np.linalg.pinv(V)) + np.trace(np.linalg.pinv(V)) + logdet_spd(V) - dY)
    return max(0.0,float(val))

def delta_g_pid(M,X,Y, rank_transform=True, bias_correct=True, max_iter=2500):
    # M=message/target; X=ELC source; Y=eco source usually.
    M=_as2d(M); X=_as2d(X); Y=_as2d(Y); n=M.shape[0]
    Mw,Xw,Yw,HX,HY=preprocess(M,X,Y,rank_transform=rank_transform)
    dM,dX,dY=Mw.shape[1],Xw.shape[1],Yw.shape[1]
    I_MX=gaussian_mi(Mw,Xw)
    I_MY=gaussian_mi(Mw,Yw)
    I_MXY=gaussian_mi(Mw,np.hstack([Xw,Yw]))
    # deficiencies: unique Y relative X, and unique X relative Y
    delta_Y_given_X=_deficiency_one(HX, HY, max_iter=max_iter)/np.log(2)
    delta_X_given_Y=_deficiency_one(HY, HX, max_iter=max_iter)/np.log(2)
    if bias_correct:
        I_MX -= bias_mi_bits(dM,dX,n)
        I_MY -= bias_mi_bits(dM,dY,n)
        I_MXY -= bias_mi_bits(dM,dX+dY,n)
        # do not bias-correct deficiencies; paper's exact bias for deficiencies is not specified in v3.
    RI=min(I_MX-delta_X_given_Y, I_MY-delta_Y_given_X)
    RI=max(0.0,RI)
    UI_X=max(0.0,I_MX-RI)
    UI_Y=max(0.0,I_MY-RI)
    SI=max(0.0,I_MXY-UI_X-UI_Y-RI)
    return {'RI':RI,'UI_X':UI_X,'UI_Y':UI_Y,'SI':SI,
            'I_MX':I_MX,'I_MY':I_MY,'I_MXY':I_MXY,
            'delta_X_given_Y':delta_X_given_Y,'delta_Y_given_X':delta_Y_given_X,
            'n':n,'dims':(dM,dX,dY),'rank_transform':rank_transform,'bias_corrected':bias_correct}
