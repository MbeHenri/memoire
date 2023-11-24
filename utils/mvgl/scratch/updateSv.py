from numpy import maximum, zeros
from numpy.linalg import norm
from .optimisation import EProgSimplex_new

def updateSv(P, beta):

    N = P.shape[0]
    gj = zeros(N, dtype=float)
    S = zeros((N, N))

    for j in range(N):
        for i in range(N):
            gj[i] = norm(P[i] - P[j])**2
            
        gj = maximum(gj.real, 0)

        # on recupere les vecteurs booléans qui indique les noeuds
        # à prendre en compte
        # on considère tous les noeuds si handle_only_k_top_neighbour = False
        # idv = [True for _ in range(N)]
        # if handle_only_k_top_neighbour:
        #     for i in np.argsort(gj)[1:(k+1)]:
        #         idv[i] = False

        b = -(gj / 2 / beta)
        S[:, j] = EProgSimplex_new(b)

    return S
