from numpy import diag, sum, maximum
from .utils import eigs

def updateIndicator(S, k):
    N = S.shape[0]
    Z = (S + S.T) / 2
    D = diag(sum(Z, axis=1))
    L = D - Z

    L = maximum(L, L.T)
    _ ,eigen_vectors, eigen_values = eigs(L,k)

    return eigen_values, eigen_vectors