from numpy import maximum, sum, tile, array, zeros, dot, eye, sqrt, transpose, mean
from numpy.linalg import solve


def normcol(G):
    # Calcul de la norme des colonnes de G
    norms = maximum(1, sqrt(
        sum(G * G, axis=0)))

    # Division des colonnes de G par leurs normes
    # en créant une matrice G' où G'(ij) = norms(j) pour tout i
    # G_ = np.tile(norms, (G.shape[0], 1))
    return G / tile(norms, (G.shape[0], 1))

# fonction de mise à jour de W
def UpdateW(H, X, W, rho=1, taux=1, maxIters=100, epsilon=1e-6):
    W_ = array(W)
    G = array(W)
    T = zeros(W.shape, dtype=float)
    d = H.shape[0]
    rho_ = rho

    A = transpose(dot(H, H.T) + rho_ * eye(d))
    for it in range(maxIters):
        W_old = W_
        # compute W(t+1)
        B = transpose(dot(X, H.T) + rho_ * (G - T))
        W_ = transpose(solve(A, B))
        # compute G(t+1)
        G = normcol(W_ + T)
        # compute T(t+1)
        T += W_ - G
        # update rho_
        rho_ = rho_ * taux
        error = mean(mean((W_ - W_old)**2))

        if it > 0 and error < epsilon:
            break
    return W_
