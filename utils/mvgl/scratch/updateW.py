from numpy import eye, dot , trace, zeros, ones
from numpy.linalg import solve

# permet de mettre à jour les poids pour réaliser la fusion des graphes des vues
def updateW(A, S, eps=2.2204e-16):

    N = A.shape[0]
    V = S.shape[0]

    Zj = zeros((N, V))
    W = zeros((N, V))
    ones_ = ones(V)

    for j in range(N):

        for v in range(V):
            Zj[:, v] = A[:, j] * S[v, :, j]

        C = dot(Zj.T, Zj)
        # regularisation de tikhonov 
        # objectif : éviter que C soit une matrice singulière (non inversible)
        C = C + eye(V) * eps * trace(C)
        # inv = np.linalg.inv(C)
        inv = solve(C, ones_)
        # W[j] = np.dot(inv, ones) / np.dot(ones, np.dot(inv, ones))
        inv = dot(inv, ones_)
        W[j] = inv / dot(ones_, inv)

    return W