from numpy import ones, zeros, sum

from .utils import have_k_components
from .updateIndicator import updateIndicator
from .updateA import updateA
from .updateW import updateW

def learningA(S, k, max_iters=30, epsilon=1e-3, verbose=True, gamma = 1):
    V = S.shape[0]
    N = S.shape[1]

    # initialisation
    W = ones((N, V))/V
    A = zeros((N, N))
    for j in range(N):
        for v in range(V):
            A[:, j] += W[j, v] * S[v, :, j]
    _, P = updateIndicator(A, k)
    A0 = A

    obj_old = 1
    obj = 10
    objectives = []
    it = 0
    while abs((obj - obj_old)/obj_old) > epsilon:
        it = it + 1

        gamma_ = gamma
        for _ in range(max_iters):
            # > update A
            A = updateA(A0, P, gamma_)

            # > update P
            P_old = P
            evs, P = updateIndicator(A, k)

            # > verification de la convergence
            ok, taux = have_k_components(evs, k)

            if ok:
                break
            else:
                gamma_ = gamma_ * taux
                if taux < 1:
                    P = P_old

        # > update W
        W = updateW(A, S)

        # calcul de la fonction objective
        obj_old = obj
        for j in range(N):
            A0[:, j] = 0
            for v in range(V):
                A0[:, j] += W[j, v] * S[v, :, j]
        obj = sum((A-A0)**2)
        # np.trace(np.dot(np.dot(P.T, L), P)) = 0 car on a déja k composantes connexes)
        objectives.append(obj)
        
        if verbose:
            print("iterations {} : Objective function : {}".format(it, obj))

    return objectives, A
