from utils.mcles.scratch.mcles import mcles
from utils.mvgl.scratch.mvgl import mvgl
from utils.vote import vote
from numpy import array, ones, unique, nan

from utils.gnps.gnps import load_data_gnps, gnps, gnps_clusters
from utils.madbyte.madbyte import preprocessing_madbyte, madbyte, load_network_madbyte, load_rmn_data_for_ml, madbyte_clusters
from utils.preprocessingml import preprocessing_for_mc

import random

def bagging(k, pathdirRMN ="rmn", pathdirMS="ms", pathdirTemp="temp", typeweak="mcles"):
    
    # chargement des données MS de molécules (branche 1)
    data_gnps = load_data_gnps()
    
    # chargement des données RMN de molécules (branche 2)
    temp_dir_rmn = f"{pathdirTemp}/madbyte"
    # prétraitement et calcul des systèmes de spin de madbyte
    preprocessing_madbyte(pathdirRMN, temp_dir_rmn)
    # chargment proprement dit (branche 2.1)
    data_madbyte = load_rmn_data_for_ml(pathdirRMN, temp_dir_rmn)
    
    # construction du reseau de gnps (branche 1)
    network_gnps = gnps(data_gnps)
    # construction du reseau de madbyte (branche 2.2)
    madbyte(temp_dir_rmn)
    network_madbyte = load_network_madbyte(temp_dir_rmn)
    
    # pretraitement des données MS et RMN de molecules (branche 2.1 + branche 1)
    dataforml = preprocessing_for_mc(data_gnps, data_madbyte)
    
    # construction des etiquetes issues de madbyte et gnps
    labels_gnps = gnps_clusters(network_gnps, dataforml["names"])
    labels_madbyte = madbyte_clusters(network_madbyte, dataforml["names"])
    params_gnps = None
    params_madbyte = None
    if labels_gnps != None:
        params_gnps = {"labels": labels_gnps}
    if labels_madbyte != None:
        params_madbyte = {"labels": labels_madbyte}
        
    # calcul des etiquetes de clusters
    labels = bagging_prime(dataforml["X"], k, typeweak=typeweak, params_gnps=params_gnps, params_madbyte=params_madbyte)
    
    return network_gnps, network_madbyte, labels

# tirage aléatoire "n" fois avec remise dans un ensemble de "n" observations pour un ensemble d'observation à regrouper
def tirage(n):
    elements = list(range(n))
    tirage = random.choices(elements, k=n)
    return unique(tirage).tolist()

# echantillonage des observatiions
def echantillonage(X):
    N = X[0].shape[1]
    V = len(X)

    ids = tirage(N)

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


def bagging_prime(X, k, nbreweak=10, typeweak="mcles", weightweak=1, params_gnps=None, params_madbyte=None):

    N = X[0].shape[1]

    labels_weaks = []
    for id in range(nbreweak):
        # echantillonage des observations
        X_ = X
        if id > 0:
            X_, ids_obs = echantillonage(X)
        
        # exécution du modèle de base
        if typeweak == "mcles":
            labels = mcles(X_, k)["labels"]
        if typeweak == "mvgl":
            labels = mvgl(X_, k)["labels"]
        
        # reconstruction et ajout des labels de clusters fournis par du modele de base
        if id > 0:
            labels = modif_labels_echantillon(ids_obs, labels, N)
        
        labels_weaks.append(labels)
    
    # ajout des labels d'étiquetes de madbyte et gnps
    if params_gnps != None:
        labels_weaks.append(params_gnps["labels"])

    if params_madbyte != None:
        labels_weaks.append(params_madbyte["labels"])

    # vote de clusters
    return vote(array(labels_weaks), ones(nbreweak), k)
