from numpy import array, ones, unique, nan, log, exp
import random

from utils.mcles.scratch.mcles import mcles
from utils.mvgl.scratch.mvgl import mvgl
from utils.utils import attach_names_on_labels
from utils.vote import vote
from .process_madbyte_gnps import process_madbyte_gnps


def bootsing(k, pathdirRMN ="rmn", pathdirMS="ms", output_dir="temp", typeweak="mcles"):
    
    # execution des processus liés à madbyte et à GNPS
    network_gnps, networks_madbyte, dataforml, params_gnps, params_madbyte = process_madbyte_gnps(pathdirRMN=pathdirRMN, pathdirMS=pathdirMS, output_dir= output_dir)
    
    # concaténation des étiquetes de madbyte et de GNPS
    labels_oracle = []
    if params_gnps != None:
        labels_oracle.append(params_gnps["labels"])

    if params_madbyte != None:
        labels_oracle.append(params_madbyte["labels"])
    
    # calcul des etiquetes de clusters
    labels = bootsing_prime(dataforml["X"], k, labels_oracle, typeweak=typeweak)
    
    # attache des noms de molécules aux étiquetes de clusters
    labels_with_names = attach_names_on_labels(dataforml["names"], labels)
    
    return network_gnps, networks_madbyte, labels_with_names

def tirage(n, weights):
    elements = list(range(n))
    tirage = random.choices(elements, weights=weights, k=n)
    return unique(tirage).tolist()

# echantillonage des observations
def echantillonage(X, weights):
    N = X[0].shape[1]
    V = len(X)

    ids = tirage(N, weights)

    X_ = []
    for v in range(V):
        X_.append(array([X[v][:, id] for id in ids]).T)

    return X_, ids

# reconstruction des etiquetes des observations echantillonées
def modif_labels_echantillon(ids, labels, len_obs):

    labs = [nan for _ in range(len_obs)]
    for id in range(len(labels)):
        labs[ids[id]] = labels[id]

    return array(labs)


def verdict_bien_classe(labels, labels_oracle):
    pass


def bootsing_prime(X, k, labels_oracle, nbreweak=10, typeweak="mcles"):
    N = X[0].shape[1]
    
    labels_weaks = []
    importances = []
    weights = ones(N)/N
    for _ in range(nbreweak):

        # échantillonage des observations
        N = X[0].shape[1]
        X_ = X
        if id > 0:
            X_, ids_obs = echantillonage(X, weights)

        # exécution du modèle de base
        if typeweak == "mcles":
            labels = mcles(X_, k, nInitForKmeans=1)["labels"]
        if typeweak == "mvgl":
            labels = mvgl(X_, k)["labels"]

        # reconstruction et ajout des labels de clusters fournis par du modele de base
        if id > 0:
            labels = modif_labels_echantillon(ids_obs, labels, N)

        # étiquetage des observations en bien classé (0) ou en mal classé (1)
        verdicts = verdict_bien_classe(labels, labels_oracle)

        # calcul de l'érreur du clustering
        error = sum(weights * verdicts)

        # calcul de l'importance du modèle
        importance = 0.5 * log((1-error)/error)

        # mise à jour des poids
        verdicts[verdicts == 0] = -1
        weights = weights * exp(importance*verdicts)
        weights = weights / sum(weights)

        importances.append(importance)
        labels_weaks.append(labels)

    return vote(array(labels_weaks), k, importances)
