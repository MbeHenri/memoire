from numpy import zeros
from .updateSv import updateSv
from .utils import have_k_components
from .updateIndicator import updateIndicator

def learningSv(X, k, max_iters_S=30, beta=1):
    # initialisation
    Q = X.T
    N = Q.shape[0]
    S = zeros((N, N))
    beta_ = beta

    for _ in range(max_iters_S):
        # > update S
        S = updateSv(Q, beta_)

        # > update Q
        Q_old = Q
        evs, Q = updateIndicator(S, k)

        # > vérification de la convergence
        ok, taux = have_k_components(evs, k)
        if ok:
            break
        else:
            beta_ = beta_ / taux
            if taux < 1:
                Q = Q_old

    return S
    
def learningS(X, k, beta=1):
    # nombre de vues
    V = len(X)
    # nombre d'observations
    N = X[0].shape[1]

    S = zeros((V, N, N))
    for v in range(V):
        S[v] = learningSv(X[v], k, beta=beta)
    return S
