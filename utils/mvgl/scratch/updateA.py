from numpy import zeros, maximum
from numpy.linalg import norm

from .optimisation import EProgSimplex_new

def updateA(A0, P, gamma):
    # A0 est la matrice de similarité issu de l'application des poids de fusion sur les
    # graphes de chacunes des vues
    
    # on vérifie si A0 possède k composantes connexes
    # si oui on retourne A0
    # sinon on calcule A
    # objectif: gagner en temps pour la résolution du problème
    # evs, _ = updateIndicator(A0, k)
    # ok, _ = have_k_components(evs, k)
    # if ok:
    #     return A0

    N = P.shape[0]
    hj = zeros(N, dtype=float)
    A = zeros((N, N))

    for j in range(N):
        for i in range(N):
            hj[i] = norm(P[i] - P[j])**2
        
        hj = maximum(hj.real, 0)
        # on recupere les vecteurs booléans qui indique les noeuds
        # à prendre en compte
        # on considère tous les noeuds si handle_only_neighbour = False
        # idv = [True for _ in range(N)]
        # if handle_only_neighbour:
        #     idv = A0[:, j] > 0

        b = -(gamma * hj * 0.5) + A0[:, j]
        A[:, j] = EProgSimplex_new(b)

    return A
