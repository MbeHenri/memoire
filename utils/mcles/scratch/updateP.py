from numpy import sum, diag
from .utils import eigs

# permet de mettre à jour P
def updateP(S, k):
    N = S.shape[0]
    Z = (S + S.T) / 2
    D = diag(sum(Z, axis=1))
    L = D - Z

    #L = np.maximum(L, L.T)
    _, P, _ = eigs(L, k)
    return L, P
