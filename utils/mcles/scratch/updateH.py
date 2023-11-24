from numpy import dot, eye
from scipy.linalg import solve_sylvester

# fonction de mise à jour de H
def UpdateH(X, W, S, alpha):
    A = dot(W.T, W)
    B = eye(S.shape[0]) - S
    B = alpha * dot(B, B.T)
    C = dot(W.T, X)
    return solve_sylvester(A, B, C)