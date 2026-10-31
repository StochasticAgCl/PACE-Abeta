"""Continuous hinge regression with a training-only minimum-SSE knot."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

def profile_sse(x, y, candidates):
    """OLS profile via sorted sufficient statistics, with no n-by-grid matrix."""
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    c = np.asarray(candidates, dtype=float)
    idx = np.searchsorted(x, c, side='right')
    suffix = lambda a: np.r_[np.cumsum(a[::-1])[::-1], 0.][idx]
    nright = len(x)-idx
    sx, sxx, sy, sxy = suffix(x), suffix(x*x), suffix(y), suffix(x*y)
    sh, sxh = sx-c*nright, sxx-c*sx
    shh, syh = sxx-2*c*sx+c*c*nright, sxy-c*sy
    normal = np.empty((len(c),3,3))
    normal[:,0,0] = len(x)
    normal[:,0,1] = normal[:,1,0] = x.sum()
    normal[:,1,1] = x@x
    normal[:,0,2] = normal[:,2,0] = sh
    normal[:,1,2] = normal[:,2,1] = sxh
    normal[:,2,2] = shh
    rhs = np.column_stack([np.full(len(c),y.sum()),np.full(len(c),x@y),syh])
    coef = np.linalg.solve(normal, rhs[...,None])[...,0]
    sse = y@y-np.einsum('ij,ij->i',coef,rhs)
    return sse,coef

def best_breakpoint(x, y, n_grid=2001):
    x, y = np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 10 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Breakpoint fitting requires at least ten finite paired observations.")
    if len(np.unique(x)) < 4 or np.ptp(y) == 0:
        raise ValueError("Insufficient variation for a continuous piecewise fit.")
    ordered = np.sort(x)
    lo = ordered[int(np.floor(len(x)*.1))]
    hi = ordered[int(np.ceil(len(x)*.9))-1]
    grid = np.linspace(lo,hi,n_grid)
    sse,_ = profile_sse(x,y,grid)
    local = np.flatnonzero((sse[1:-1]<=sse[:-2])&(sse[1:-1]<=sse[2:]))+1
    trials = [(float(grid[i]),float(sse[i])) for i in [0,len(grid)-1,int(sse.argmin())]]
    for i in local:
        fit = minimize_scalar(lambda c: profile_sse(x,y,[c])[0][0],
                              bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-11})
        trials.append((float(fit.x),float(fit.fun)))
    bp,minimum = min(trials,key=lambda item:item[1])
    # Recompute directly to avoid cancellation affecting reported fit statistics.
    design = np.column_stack([np.ones(len(x)),x,np.maximum(x-bp,0)])
    beta = np.linalg.lstsq(design,y,rcond=None)[0]
    direct = np.square(y-design@beta).sum()
    np.testing.assert_allclose(minimum,direct,rtol=1e-9,atol=1e-6)
    return bp,pd.DataFrame(dict(breakpoint=grid,sse=sse)),(lo,hi)


def describe_piecewise(x, y, knot):
    """Fit a descriptive curve at a fixed knot; segment tests never select it."""
    import statsmodels.api as sm
    x, y = np.asarray(x, float), np.asarray(y, float)
    design = np.column_stack([np.ones(len(x)), x, np.maximum(x-knot, 0)])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    result = {"knot": float(knot), "coefficients": coefficients.tolist(),
              "left_CL_per_01": float(coefficients[1]/10),
              "right_CL_per_01": float((coefficients[1]+coefficients[2])/10),
              "r2": float(1-np.sum((y-design@coefficients)**2)/np.sum((y-y.mean())**2))}
    for name, mask in [("left", x <= knot), ("right", x > knot)]:
        if mask.sum() < 3 or len(np.unique(x[mask])) < 2:
            raise ValueError("The fixed knot leaves too few distinct observations in a segment.")
        model = sm.OLS(y[mask], sm.add_constant(x[mask])).fit()
        result[name] = {"n": int(mask.sum()), "p": float(model.pvalues[1]), "r2": float(model.rsquared)}
    return result
