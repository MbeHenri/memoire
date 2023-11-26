import os
from source.core import construct_spin_system, construct_correlation_matrix, create_outputs
import networkx as nx
import pandas as pd
import json
from numpy import zeros, nan

#> prétraitement et calcul des systèmes de spin de madbyte
def preprocessing_madbyte(input_dir, output_dir):
    try:
        os.mkdir(f'{output_dir}')
    except FileExistsError:
        pass

    list_samples = []
    for item in os.listdir(input_dir):
        list_samples.append(item)
        try:
            os.mkdir(f'{output_dir}/{item}')
        except FileExistsError:
            pass

    # construction des ensembles de déplacements(système de spin) par molécule
    for name in list_samples:
        construct_spin_system(
            name, input_dir, f'{output_dir}/{name}', nmr_data_type="CSV")

#> construction des reseaux moléculaires de madbyte
def madbyte(output_dir):
    # construction de la matrice de correlation entre les ensembles de déplacmements
    construct_correlation_matrix(output_dir)

    # construction des reseaux moleculaires
    create_outputs(output_dir, fname="madbyte")


#> chargement du reseau moleculaire de madbyte
def load_network_madbyte(output_dir, type="hybrid"):
    G = None
    if type == "all":
        G = nx.read_graphml(
            f"{output_dir}/madbyte_association_network_all.graphml")
    if type == "similarity":
        try:
            G = nx.read_graphml(
                f"{output_dir}/madbyte_similarity_network_network.graphml")
        except:
            pass
    if type == "hybrid":
        try:
            G = nx.read_graphml(f"{output_dir}/madbyte_hybrid_network.graphml")
        except:
            pass
    return G

def combined_spin_system_for_sample(hsqc_toscy):
    deplacements = []
    for deplacement in hsqc_toscy.values():
        deplacements = deplacements + deplacement 
    return deplacements

# > chargement des spectres rmn issues du pretraitement de madbyte 
# pour l'appentissage artificielle
def load_rmn_data_for_ml(input_dir, output_dir):

    # chargement des groupes de déplacements chimiques de molécules
    spectres = []
    for name in os.listdir(input_dir):
        spectre = {}
        spectre["name"] = name
        hsqc_toscy = ""
        with open(f'{output_dir}/{name}/{name}_spin_systems.json', 'r') as f:
            for ligne in f:
                hsqc_toscy = f'{hsqc_toscy}{ligne}'
        spectre["hsqc_toscy"] = hsqc_toscy
        spectres.append(spectre)
        
    # construction de dataframe des spectres en regroupant pour chaque molécule, tous les déplacements chimiques
    df_2drmn = pd.DataFrame(spectres)
    #df_2drmn.to_json('{output_dir}/hsqc_tocsy_coupled.json')
    df_2drmn["hsqc_toscy"] = df_2drmn["hsqc_toscy"].apply(json.loads)
    df_2drmn["hsqc_toscy"] = df_2drmn["hsqc_toscy"].apply(combined_spin_system_for_sample)
    df_2drmn.to_json('hsqc_tocsy_ml.json')
    return df_2drmn
    
# > construction des clusters à partir du réseau moléculaire de madbyte
def madbyte_clusters(G, names_mols):
    # calcul des composantes connexes du réseau qui constituent les clusters de molécules
    clusters_mol = {}
    k=0
    for component in nx.connected_components(G):
        for node in component:
            if G.nodes[node]["_type"] == "standard":
                clusters_mol[node] = k
        k = k + 1
    
    n = len(names_mols)
    mols = zeros(n)
    for i in range(n):
        try:
            mols[i] = clusters_mol[names_mols[i]]
        except :
            mols[i] = nan
            
    return mols