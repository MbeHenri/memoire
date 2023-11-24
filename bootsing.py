from utils.mcles.scratch.mcles import mcles
from utils.mvgl.scratch.mvgl import mvgl

from utils.vote import vote
from numpy import array, ones, zeros


def oracle(labels_oracles, mol1, mol2):
    pass


def bootsing(k, nbreweaks=10, typeweaks="mcles"):
    pass

#
def selection(X, weights):
    pass

#
def calcul_erreur(labels, labels_oracles):
    pass

#
def calcul_importance(erreur):
    pass

#
def modif_labels_echantillon(ids, labels, len_obs):
    pass

#
def modif_weights(weights, erreur):
    pass

def bootsing_prime(X, k, nbreweak=10, typeweak="mcles", labels_oracles=None):
    N = X[0].shape[1]
    if typeweak is "mcles":

        labels_weaks = []
        weights = ones(N)/N
        importances = zeros(nbreweak)

        for it in range(nbreweak):
            # echantillonage
            X_, ids_obs = selection(X, weights)

            labels = mcles(X_, k)["labels"]

            labels = modif_labels_echantillon(ids_obs, labels, N)

            error = calcul_erreur(labels, labels_oracles)
            
            weights = modif_weights(weights, error)
            
            importances[it] = calcul_importance(error)

            labels_weaks.append(labels)

        return vote(array(labels), importances, k)
