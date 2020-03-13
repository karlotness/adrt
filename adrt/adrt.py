r"""
Approximate Discrete Radon Transform (ADRT)
see LICENSE
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from . import _adrtc


def div0(a,b):
    r"""
    avoid divide by zero for arrays

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def f(x,*args):
    r"""
    computes ||Rx - B||**2
    x is expected to be a flattend array.

    R : forward DRT _adrt.adrtm.pdrtaq(,)

    """
    N = x.shape[0]
    n = int(np.sqrt(N))
    x2 = x.reshape((n,n))

    pdaa = args[0]
    pdab = args[1]
    pdac = args[2]
    pdad = args[3]
    p = args[5]

    pdaax,pdabx,pdacx,pdadx = _adrt.adrtm.pdrtaq(x2,p)


    val = 0.
    val = val + ((pdaa - pdaax) ** 2).sum()
    val = val + ((pdab - pdabx) ** 2).sum()
    val = val + ((pdac - pdacx) ** 2).sum()
    val = val + ((pdad - pdadx) ** 2).sum()

    return val


def gradf(x, *args):
    r"""
    computes 2*(Rx - B)

    """
    bab = args[4]
    p = args[5]

    N = x.shape[0]
    n = int(np.round(np.sqrt(N)))
    x2 = x.reshape((n,n))

    ba = _adrt.adrtm.pmatmul(x2,p)

    return 2.*(ba - bab).flatten()


def matmul(x, p=1):
    r"""
    computes R R^# with prologation p

    :INPUT:

    x is a 1D array (flattened 2D array)
    p kwarg is the prologation level, should be a power of 2

    TODO: assert that p is a power of 2

    """

    N = x.shape[0]
    n = int(np.round(np.sqrt(N)))
    x2 = x.reshape((n,n))

    #ba = _adrt.adrtm.pmatmul(x2,p)

    da = _adrtc.adrt(x2)
    baa,bab,bac,bad = _adrtc.bdrt(da[0],da[1],da[2],da[3])
    ba = baa + bab + bac + bad

    return ba



def matmulw(x,p=1):
    r"""
    computes R* R = R# W R where W is the weights w(h), with prologation p

    :INPUT:

    x is a 1D array (flattened 2D array)
    p is the prologation level, should be a power of 2

    TODO: this should be written in the fortran module

    """

    N2 = x.shape[0]
    N = int(np.round(np.sqrt(N2)))
    x2 = x.reshape((N,N))
    pN = p*N

    #ba = _adrt.adrtm.pmatmulw(x2,p)   # TODO: write this

    pdaa,pdab,pdac,pdad = _adrt.adrtm.pdrtaq(x2,p)

    M = pdaa.shape[1]

    for j in range(M):

        # compute weight function w(s) at the mid-points
        ss = np.linspace(-1.,1.,N+j+1)
        ss = .5*(ss[0:-1] + ss[1:])
        ws = 1. / np.sqrt(1 - ss**2)

        # weight transformed arrays
        pdaa[(pN-j):(2*pN),j] *= ws
        pdab[(pN-j):(2*pN),j] *= ws
        pdac[(pN-j):(2*pN),j] *= ws
        pdad[(pN-j):(2*pN),j] *= ws

    #costh = np.cos(np.linspace(0.,np.pi/4.,M))

    #pdaa *= costh
    #pdab *= costh
    #pdac *= costh
    #pdad *= costh

    ba = _adrt.adrtm.rbdrtaq(pdaa,pdab,pdac,pdad,p)

    return ba

def adrt(a,p=1):
    r"""
        compute adrt

    """

    da = _adrtc.adrt(a)

    return da


def badrt(da,r=1):
    r"""
        compute back projection of adrt

    """

    daa = da[0]
    dab = da[1]
    dac = da[2]
    dad = da[3]

    ba = _adrt.adrtm.rbdrtaq(daa,dab,dac,dad,r)

    return ba


def compute_dlines(N,h,s,q,r=1,return_binary=True):
    r"""

    Returns a 2D array of size N x N whose value is 1.0 if the pixel lies on
    the specified d-line and 0 elsewhere.

    Parameters
    ----------

    N: image size
    s: slope
    h: intersect, ranges from -s to N
    q: quadrant

    """
    
    daa = []
    for quad in range(4):
         daq = np.zeros((3*N,N))
         daa.append(daq)

    daa[q][h+N-1,s] = float(N**3)
    ba = badrt(daa,r)
    if return_binary:
        ba = (ba > 1e-8)
    
    return ba

def compute_dlines2(N,h,s,q,r=1):

    daa = []
    for quad in range(4):
         daq = np.zeros((3*N,N))
         daa.append(daq)

    daa[q][h,s] = float(N**3)
    ba = badrt(daa,r)
    ba = 1.*(ba > 1e-8)
    ba /= np.sum(ba.flatten())

    return ba

def plot(da,masked=True,**kwargs):
    r"""

    plot four-quadrant ADRT 


    """
    keys = kwargs.keys()

    plot_type=kwargs['plot_type'] if 'plot_type' in keys else 'pcolor'
    fs=kwargs['figsize'] if 'figsize' in keys else (8.0,4.5)
    ec=kwargs['edgecolor'] if 'edgecolor' in keys else None
    lw=kwargs['linewidth'] if 'linewidth' in keys else 0.1

    # label quadrants a,b,c,d
    daa = da[0]
    dab = da[1]
    dac = da[2]
    dad = da[3]

    vmin = np.min([daa.min(),dab.min(),dac.min(),dad.min()])
    vmax = np.max([daa.max(),dab.max(),dac.max(),dad.max()])
    vabs = np.max([abs(vmin),abs(vmax)])
    
    if (vmin >= 0.) :
        cm=kwargs['cmap'] if 'cmap' in keys else 'Reds'
        levels = np.linspace(0.0,vmax,25)
        vmin = 0.0
        vmax = vmax
    elif (vmax <= 0.0) :
        cm=kwargs['cmap'] if 'cmap' in keys else 'Blues_r'
        levels = np.linspace(vmin,0.0,25)
        vmin = vmin
        vmax = 0.0
    else:
        cm=kwargs['cmap'] if 'cmap' in keys else 'RdBu_r'
        levels = np.linspace(-vabs,vabs,25)
        vmin = -vabs
        vmax =  vabs

    if 'vmax' in keys:
        vmax = kwargs['vmax']
        vmin = -vmax

    N = daa.shape[1]

    if (plot_type == 'pcolor'):
        m0 = 1
    elif (plot_type == 'contourf'):
        m0 = 0

    x0 = np.arange(N+m0)
    x = np.linspace(0,np.pi/4.0,N+m0)
    y = np.arange(2*N+m0)
    X,Y = np.meshgrid(x,y)
    Y = Y-N+1   # why?

    mask0 = np.ones((N,N),dtype='bool')

    fig0 = plt.figure(figsize=fs)
    ax0 = fig0.add_axes([0.10,0.15,0.95,0.70])
    
    masknot = np.concatenate((np.tril(mask0)[:,::-1], mask0), axis=0)
    mask1 = np.bitwise_not(masknot)

    quadrant_bdry_array = np.linspace(0,np.pi,5)
    quadrant_bdry_array = quadrant_bdry_array[1:-1] 

    for bdry in quadrant_bdry_array:
        plt.plot([bdry,bdry],[-1-2*N,N+1],'k:')

    daa_masked = np.ma.masked_array(daa[    :2*N,:],mask=mask1)
    dab_masked = np.ma.masked_array(dab[2*N:0:-1,:],mask=mask1[::-1,:])
    dac_masked = np.ma.masked_array(dac[    :2*N,:],mask=mask1)
    dad_masked = np.ma.masked_array(dad[2*N:0:-1,:],mask=mask1[::-1,:])

    if (plot_type == 'contourf'):
        ax0.contourf(X,Y,daa_masked,cmap=cm,levels=levels,extend='both')
        ax0.contourf(X[:,::-1] + np.pi/4.0 , Y,
                     dab_masked,cmap=cm,levels=levels,extend='both')
        ax0.contourf(X+np.pi/2.0, Y-N+1,
                     dac_masked,cmap=cm,levels=levels,extend='both')
        im0 =\
        ax0.contourf(X[:,::-1] + np.pi*3.0/4.0 , Y-N+1,
                     dad_masked,cmap=cm,levels=levels,extend='both')
        ax0.set_xlim([   0.0, np.pi])
        ax0.set_ylim([-1-2*N,     N])

    elif (plot_type == 'pcolor'):
        ax0.pcolormesh(X,Y,daa_masked,\
                       vmin=vmin,vmax=vmax,cmap=cm,edgecolor=ec,linewidth=lw)
        ax0.pcolormesh(X[:,::-1]+np.pi/4.0, Y, dab_masked,
                       vmin=vmin,vmax=vmax,cmap=cm,edgecolor=ec,linewidth=lw)
        ax0.pcolormesh(X+np.pi/2.0, Y-N+1, dac_masked,\
                       vmin=vmin,vmax=vmax,cmap=cm,edgecolor=ec,linewidth=lw)
        im0 = \
        ax0.pcolormesh(X[:,::-1]+np.pi*3.0/4.0, Y-N+1, dad_masked,\
                       vmin=vmin,vmax=vmax,cmap=cm,edgecolor=ec,linewidth=lw)
        ax0.set_xlim([   0.0, np.pi])
        ax0.set_ylim([-2*N+2,   N+1])

    # add quadrant annotations
    ax0.annotate(xy=(  np.pi/8.0,-3*N/2), s="a")
    ax0.annotate(xy=(np.pi*3/8.0,-3*N/2), s="b")
    ax0.annotate(xy=(np.pi*5/8.0,   N/2), s="c")
    ax0.annotate(xy=(np.pi*7/8.0,   N/2), s="d")

    # set x-axis ticks
    dt = np.diff(np.arctan([0.0,0.5,1.0]))
    xticks = np.cumsum([0.0] + [dt[j%2] for j in range(1,9)])
    xticklabels_th = [ "$0$", r"$\frac{1}{8}\pi$", r"$\frac{1}{4}\pi$",\
         r"$\frac{3}{8}\pi$", r"$\frac{1}{2}\pi$", r"$\frac{5}{8}\pi$",\
         r"$\frac{3}{4}\pi$", r"$\frac{7}{8}\pi$", r"$\pi$"]
    ax0.set_xticks(xticks)
    ax0.set_xticklabels(xticklabels_th)

    # add second x-axis
    ax2 = ax0.twiny()
    xticks = np.linspace(0.0,np.pi,9)
    xticklabels_s = ["$0$", r"$\frac{N}{2}$", r"$N$", r"$\frac{N}{2}$",\
                     "$0$", r"$\frac{N}{2}$", r"$N$", r"$\frac{N}{2}$","$0$"]
    ax2.set_xlabel("$s$")
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels_s)

    ax0.set_xlabel(r"$\theta$")
    ax0.set_ylabel("$h$")

    # add colorbar
    fig0.colorbar(im0,ax=ax0)
    
    return fig0


def density(N,p):
    r"""

    return DRT density drho

    """
    rho0 = np.ones((N,N))
    drho = _adrt.adrtm.pdrt(rho0,p)
    drho = drho / drho.max()

    return drho


def pdrtaqd(a,p):
    r"""
        pdrtaq with density correction

    """

    N = a.shape[0]

    # compute density
    drho = density(N,p)

    pdaa,pdab,pdac,pdad = _adrt.adrtm.pdrtaq(a,p)

    pdaa = div0(pdaa,drho)
    pdab = div0(pdab,drho)
    pdac = div0(pdac,drho)
    pdad = div0(pdad,drho)

    return pdaa,pdab,pdac,pdad


def iadrt(da,p=1,x0=None,tol=1e-8,maxiter=1000):
    r"""
        inverse ADRT

    """

    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse.linalg import cg

    N = int(da[0].shape[1] / p)
    ba = _adrtc.bdrt(da[0],da[1],da[2],da[3])
    b = ba[0] + ba[1] + ba[2] + ba[3]
    b = b.flatten()
    if (type(x0) == type(None)):
        x0 = b.copy()

    N1 = int(b.shape[0]*(p**2))

    def matmul_p(x):
        Ax = matmul(x)
        return Ax.flatten()

    A = LinearOperator((N**2,N**2),matvec=matmul_p,dtype=type(0.))
    out = cg(A,b,tol=tol,x0=x0,maxiter=maxiter)

    x2 = out[0].reshape(N,N)

    return x2

