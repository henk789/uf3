import ndsplines
import numpy as np

np.random.seed(123)


'''
from the ndsplines package
slightly simplyfied for our usecase
knots from 0 to 1
'''
#def make_random_spline(xdim=1, k=None, periodic=False, extrapolate=True, yshape=None, ydim=1, ymax=10):
def make_random_spline(xdim=1, k=None, periodic=False, extrapolate=False):
    ns = []
    ts = []
    if k is None:
        ks = np.random.randint(5, size=xdim)
    else:
        ks = np.broadcast_to(k, (xdim,))
    if periodic is None:
        periodic = np.random.randint(2,size=xdim, dtype=np.bool_)
    if extrapolate is None:
        extrapolate = np.random.randint(2,size=xdim, dtype=np.bool_)

    # if ydim is None:
    #     ydim = np.random.randint(5)
    # if yshape is None:
    #     yshape = tuple(np.random.randint(1, ymax, size=ydim))
    ydim = 1
    yshape = ()
    for i in range(xdim):
        ns.append(np.random.randint(7) + 2*ks[i] + 3)
        ts.append(np.r_[0.0:0.0:(ks[i]+1)*1j,
            np.sort(np.random.rand(ns[i]-ks[i]-1)),
            1.0:1.0:(ks[i]+1)*1j
            ])
    c = np.random.rand(*ns,*yshape)
    return ndsplines.NDSpline(ts, c, ks, periodic, extrapolate)

'''
generate a random spline with equal knot spacing
knots from 0 to 1
'''
def make_random_equidistant_spline(xdim=1, k=None, edge_knots=False, periodic=False, extrapolate=False, seed=123):
    np.random.seed(seed)
    ns = []
    ts = []
    if k is None:
        ks = np.random.randint(5, size=xdim)
    else:
        ks = np.broadcast_to(k, (xdim,))
    if periodic is None:
        periodic = np.random.randint(2,size=xdim, dtype=np.bool_)
    if extrapolate is None:
        extrapolate = np.random.randint(2,size=xdim, dtype=np.bool_)

    ydim = 1
    yshape = ()
    for i in range(xdim):
        ns.append(np.random.randint(7) + 2*ks[i] + 3)
        if edge_knots:
            ts.append(np.r_[0.0:0.0:(ks[i]+1)*1j,
                (np.arange(ns[i]-ks[i]) * (1/(ns[i]-ks[i])))[1:],
                1.0:1.0:(ks[i]+1)*1j
                ])
        else:
            ts.append(np.r_[0.0,
                (np.arange(ns[i]+ks[i]) * (1/(ns[i]+ks[i])))[1:],
                1.0
                ])
    c = np.random.rand(*ns,*yshape)
    return ndsplines.NDSpline(ts, c, ks, periodic, extrapolate)


def make_random_equidistant_spline_2(xdim=1, k=None, seed=123):
    np.random.seed(seed)
    ns = []
    ts = []
    pad = []
    if k is None:
        ks = np.random.randint(5, size=xdim)
    else:
        ks = np.broadcast_to(k, (xdim,))

    ydim = 1
    yshape = ()
    for i in range(xdim):
        ns.append(np.random.randint(7) + 2*ks[i] + 3)
        ts.append(np.r_[0.0:0.0:(ks[i]+1)*1j,
            (np.arange(ns[i]-ks[i]) * (1/(ns[i]-ks[i])))[1:],
            1.0:1.0:(ks[i]+1)*1j
            ])
        ns[i] -= 2*ks[i]
        pad.append((ks[i],ks[i]))
    c = np.random.rand(*ns,*yshape)
    c = np.pad(c, pad)
    return ndsplines.NDSpline(ts, c, ks, periodic=False, extrapolate=False)