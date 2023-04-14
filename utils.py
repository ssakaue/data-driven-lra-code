from scipy import sparse
import jax.numpy as jnp
from jax import vmap, jit, grad, random

def topk_binary_matrix(X, s):
    ncol = X.shape[1]
    row = jnp.argsort(-jnp.abs(X), axis=0)[:s,:].reshape(-1)
    col = list(range(ncol))*s
    return sparse.csr_matrix((jnp.ones(ncol*s), (row, col)), shape = X.shape).toarray()

def svd_trnc(A, svd_prec):
    U, svals, Vh = jnp.linalg.svd(A, full_matrices=False)
    idx = jnp.where(jnp.abs(svals) > svd_prec)[0]
    return U[:,idx], svals[idx], Vh[idx,:]

def scw(S, A, k, svd_prec):
    SA = S@A
    r = jnp.linalg.matrix_rank(SA)
    _, _, VhS = svd_trnc(SA, svd_prec)
    VhS = VhS[:r,:]
    U, svals, Vh = svd_trnc(A@VhS.T, svd_prec)
    kr = min(k, r)
    AVk = U[:,:kr] @ jnp.diag(svals[:kr]) @ Vh[:kr,:]
    return AVk@VhS