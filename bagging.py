from numpy import array, ones, unique, nan
from joblib import Parallel, delayed
import random

from utils.mcles.scratch.mcles import mcles
from utils.mvgl.scratch.mvgl import mvgl
from utils.vote import vote
from utils.utils import attach_names_on_labels
from process_madbyte_gnps import process_madbyte_gnps


def bagging(k, pathdirRMN="rmn", pathdirMS="ms", output_dir="temp", typeweak="mcles"):
    # execution des processus liés à madbyte et à GNPS
    (
        network_gnps,
        networks_madbyte,
        dataforml,
        params_gnps,
        params_madbyte,
    ) = process_madbyte_gnps(
        pathdirRMN=pathdirRMN, pathdirMS=pathdirMS, output_dir=output_dir
    )

    # calcul des etiquetes de clusters
    labels = bagging_prime(
        dataforml["X"],
        k,
        typeweak=typeweak,
        params_gnps=params_gnps,
        params_madbyte=params_madbyte,
    )

    # attache des noms de molécules aux étiquetes de clusters
    labels_with_names = attach_names_on_labels(dataforml["names"], labels)

    return network_gnps, networks_madbyte, labels_with_names


# tirage aléatoire "n" fois avec remise dans un ensemble de "n" observations pour un ensemble d'observation à regrouper
# dans le cas où on a défini le nombre p d'observations alors on fait une selection de "p" observations
def tirage(n, p=None):
    elements = list(range(n))

    if p is None:
        tirage = random.choices(elements, k=n)
        tirage = unique(tirage).tolist()
    else:
        tirage = random.sample(elements, int(p * n))
    return tirage


# echantillonage des observations
def echantillonage(X, p=None):
    N = X[0].shape[1]
    V = len(X)

    ids = tirage(N, p=p)

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


def work_weak(X, k, id=0, typeweak="mcles", p=None, nInitForKmeans=10):
    # echantillonage des observations
    N = X[0].shape[1]

    if id == 0:
        X_ = X
    else:
        X_, ids_obs = echantillonage(X, p=p)

    # exécution du modèle de base
    if typeweak == "mcles":
        labels = mcles(X_, k, nInitForKmeans=nInitForKmeans)["labels"]
    if typeweak == "mvgl":
        labels = mvgl(X_, k)["labels"]

    # reconstruction et ajout des labels de clusters fournis par les modeles de base
    if id > 0:
        labels = modif_labels_echantillon(ids_obs, labels, N)

    return labels


def bagging_prime(
    X,
    k,
    nbreweak=10,
    typeweak="mcles",
    params_gnps=None,
    params_madbyte=None,
    p=None,
    nInitForKmeans=10,
):
    labels_weaks = Parallel(n_jobs=-1)(
        delayed(work_weak)(
            X, k, id=id, typeweak=typeweak, p=p, nInitForKmeans=nInitForKmeans
        )
        for id in range(nbreweak)
    )

    # ajout des labels d'étiquetes de madbyte et gnps
    if params_gnps is not None:
        labels_weaks.append(params_gnps["labels"])

    if params_madbyte is not None:
        labels_weaks.append(params_madbyte["labels"])

    # vote de clusters
    return vote(array(labels_weaks), k, weights_weaks=ones(nbreweak))
