from numpy import real, argsort, sort, sum
from numpy.linalg import eig

# permet de calculer le taux à appliquer aux constantes de régularizations lamda et beta
# de l'algorithme MVGL
def get_taux_regul(sum_evs, evk, epsilon=1e-11):

    taux = 1
    # on a moins de k composents connexes
    if sum_evs > epsilon:
        # on augmente l'importance de l'apprentissage des poids de clusters
        taux = 2

    # on a plus de k composents connexes
    elif sum_evs + evk < epsilon:
        # on diminue l'importance de l'apprentissage des poids de clusters
        taux = 1/2

    return taux

# permet de vérifier si un graphe à k composents sur la base des valeurs propres de son laplacien
def have_k_components(evs, k, epsilon=1e-11):
    ok = False

    # somme des k premieres valeurs propres
    evs = sort(evs.real)
    sum_propre = sum(evs[:k])
    taux = get_taux_regul(sum_propre, evs[k], epsilon=epsilon)
    
    if taux == 1:
        ok = True

    return ok, taux


# permet de retourner les k premieres valeurs propres avec leur vecteurs
# ainsi que toutes les valeurs propres en ordre
def eigs(M, k):
    eigen_values, eigen_vectors = eig(M)
    idx = argsort(real(eigen_values))
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    return eigen_values[:k].real, eigen_vectors[:, :k], eigen_values
