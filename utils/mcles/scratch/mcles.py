from numpy import sqrt, concatenate, zeros, dot, sum, trace, sqrt
from numpy.random import rand
from sklearn.cluster import KMeans
from .updateW import UpdateW
from .updateH import UpdateH
from .updateS import UpdateS
from .updateP import updateP

def mcles(X, k, d=45, maxIters=30, alpha=0.8, beta=0.5, gamma=0.004, epsilon=0.01, maxItersForKmeans=1000, nInitForKmeans=20, verbose=False):
    # nombre de vues
    V = len(X)
    # nombre d'observations
    N = X[0].shape[1]

    for i in range(V):
        # normalisation de chaque observation (on transforme les vecteurs d'observations en leurs vecteurs unitaires)
        temp = sqrt(sum(X[i]**2, axis=0))
        X[i] /= temp

    X_ = concatenate(X, axis=0)
    D = [X[i].shape[0] for i in range(V)]
    SD = sum(D)
    W = zeros((SD, d), dtype=float)
    H = rand(d, N)
    S = zeros((N, N), dtype=float)
    P = rand(N, k)

    # tableau des valeurs de la fonction objective
    objectives = []

    for it in range(maxIters):
        # > update W
        W = UpdateW(H, X_, W)
        # > update H
        H = UpdateH(X_, W, S, alpha)
        # > update S
        S = UpdateS(H, P, alpha, beta, gamma)
        # > update P
        L, P = updateP(S, k)

        # > vérification de la convergence
        #  calcul et enregistrement des valeurs de la fonction objective
        obj = sum((X_ - dot(W, H))**2) + alpha * sum((H - dot(H, S))**2) \
            + beta * sum(S**2) + gamma * \
            trace(dot(dot(P.T, L), P))
        objectives.append(obj.real)
        if verbose:
            print("iterations {} : Objective function : {}".format(it, obj))
        if it > 0 and (abs(obj - objectives[it-1])/objectives[it-1]) < epsilon:
            break

    kmeans = KMeans(
        n_clusters=k,
        n_init=nInitForKmeans,
        max_iter=maxItersForKmeans,
    )
    kmeans.fit(P.real)
    labels_clusters = kmeans.labels_

    return {"W": W, "H": H, "S": S, "P": P.real, "labels": labels_clusters, "objs": objectives}
