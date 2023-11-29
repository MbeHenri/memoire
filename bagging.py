from utils.mcles.scratch.mcles import mcles
from utils.mvgl.scratch.mvgl import mvgl
from utils.vote import vote
from numpy import array, ones, unique, nan

from utils.gnps.own import load_data_gnps, gnps, gnps_clusters
from utils.madbyte.own import preprocessing_madbyte, madbyte, load_network_madbyte, load_rmn_data_for_ml, madbyte_clusters
from utils.preprocessingml import preprocessing_for_mc
from utils.utils import attach_names_on_labels

from joblib import Parallel, delayed
from tqdm import tqdm

import random

def bagging(k, pathdirRMN ="rmn", pathdirMS="ms", output_dir="temp", typeweak="mcles"):
    
    # chargement des données MS de molécules (branche 1)
    data_gnps = load_data_gnps()
    
    # chargement des données RMN de molécules (branche 2)
    output_dir_rmn = f"{output_dir}/madbyte"
    # prétraitement et calcul des systèmes de spin de madbyte
    preprocessing_madbyte(pathdirRMN, output_dir_rmn)
    # chargment proprement dit (branche 2.1)
    data_madbyte = load_rmn_data_for_ml(pathdirRMN, output_dir_rmn)
    
    # construction du reseau de gnps (branche 1)
    network_gnps = gnps(data_gnps)
    # construction des reseaux de madbyte (branche 2.2)
    madbyte(output_dir_rmn, fname="madbyte")
    network_madbyte_all = load_network_madbyte(output_dir_rmn, name="madbyte", type="all")
    network_madbyte_sim = load_network_madbyte(output_dir_rmn, name="madbyte", type="sim")
    network_madbyte_hybrid = load_network_madbyte(output_dir_rmn, name="madbyte", type="hybrid")
    networks_madbyte = [network_madbyte_all, network_madbyte_sim, network_madbyte_hybrid]
    
    # pretraitement des données MS et RMN de molecules (branche 2.1 + branche 1)
    dataforml = preprocessing_for_mc(data_gnps, data_madbyte, output_dir)
    
    # construction des etiquetes issues de madbyte et gnps
    labels_gnps = gnps_clusters(network_gnps, dataforml["names"])
    labels_madbyte = madbyte_clusters(network_madbyte_hybrid, dataforml["names"])
    
    params_gnps = None
    if labels_gnps != None:
        params_gnps = {"labels": labels_gnps}
        
    params_madbyte = None
    if labels_madbyte != None:
        params_madbyte = {"labels": labels_madbyte}
        
    # calcul des etiquetes de clusters
    labels = bagging_prime(dataforml["X"], k, typeweak=typeweak, params_gnps=params_gnps, params_madbyte=params_madbyte)
    
    # attache des noms de molécules aux étiquetes de clusters
    labels_with_names = attach_names_on_labels(dataforml["names"], labels)
    
    return network_gnps, networks_madbyte, labels_with_names

# tirage aléatoire "n" fois avec remise dans un ensemble de "n" observations pour un ensemble d'observation à regrouper
def tirage(n):
    elements = list(range(n))
    tirage = random.choices(elements, k=n)
    return unique(tirage).tolist()

# echantillonage des observations
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


def work_weak(X, k, id=0, typeweak="mcles"):
    # echantillonage des observations
    N = X[0].shape[1]
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
        
    return labels

def bagging_prime(X, k, nbreweak=10, typeweak="mcles", weightweak=1, params_gnps=None, params_madbyte=None):

    N = X[0].shape[1]
        
    labels_weaks = Parallel(n_jobs=-1)(
        delayed(work_weak)(X,k,id=id,typeweak=typeweak) for id in range(nbreweak)
    )
    
    # ajout des labels d'étiquetes de madbyte et gnps
    if params_gnps != None:
        labels_weaks.append(params_gnps["labels"])

    if params_madbyte != None:
        labels_weaks.append(params_madbyte["labels"])

    # vote de clusters
    return vote(array(labels_weaks), ones(nbreweak), k)
