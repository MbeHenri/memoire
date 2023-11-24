from numpy import dot, eye, ones, zeros, maximum
from numpy.linalg import norm

from qpsolvers import solve_qp

# fonction de mise à jour de S
def UpdateS(H, P, alpha=0.001, beta=0.5, gamma=0.004):
    N = H.shape[1]
    bi = zeros(N, dtype=float)

    K = dot(H.T, H)
    Q = K + (beta/alpha) * eye(K.shape[0], dtype=float)
    Q = (Q + Q.T)/2

    # pour les contraintes linéaire

    A = ones((N, N), dtype=float)
    ones_ = ones(N, dtype=float)  # uh , b
    zeros_ = zeros(N, dtype=float)  # lh

    S_ = zeros((N, N))
    for i in range(N):

        # calcul de bi
        for j in range(N):
            bi[j] = norm(P[i] - P[j])**2
        
        bi = maximum(bi.real, 0)
        
        # calcul de r
        r = -2*K[i, :] + (gamma / (2*alpha))*bi

        # calcul de S:i
        Si = solve_qp(Q, r, None, None, A, ones_, zeros_, ones_, solver="osqp")
        if Si is not None:
            S_[:, i] = Si
    return S_
