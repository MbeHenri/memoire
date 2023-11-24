from numpy import real, argsort
from numpy.linalg import eig

# permet de retourner les k premieres valeurs propres avec leur vecteurs
# ainsi que toutes les valeurs propres en ordre
def eigs(M, k):
    eigen_values, eigen_vectors = eig(M)
    idx = argsort(real(eigen_values))
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    return eigen_values[:k].real, eigen_vectors[:, :k], eigen_values