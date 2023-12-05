from os import mkdir, listdir
from networkx import read_graphml, connected_components
from pandas import DataFrame
from json import loads
from numpy import zeros, nan
from networkx import bfs_tree

from .madbyte import spin_system_construction, generate_network, correlation_matrix_generation

#> prétraitement et calcul des systèmes de spin de madbyte
def preprocessing_madbyte(input_dir, output_dir, entity="Extract", solvent="DMSO-D6", nmr_data_type="CSV", hppm_error = 0.05, tocsy_error=0.05):
    try:
        mkdir(f'{output_dir}')
    except FileExistsError:
        pass
    sample_list = listdir(input_dir)
    spin_system_construction(sample_list, input_dir, nmr_data_type, entity, hppm_error, tocsy_error, output_dir, solvent=solvent)

#> construction des reseaux moléculaires de madbyte
def madbyte(output_dir, fname="madbyte", hppm_error = 0.05, cppm_error=0.5, threshold=0.5, colors = {"spin": "#d3d7cf", "extract": "#ff3333", "standard": "#0ffbff"}):
    # construction de la matrice de correlation entre les ensembles de déplacemements
    correlation_matrix_generation(hppm_error, cppm_error, output_dir)
    
    # construction des reseaux moleculaires
    generate_network(output_dir,threshold,fname,cppm_error,hppm_error, colors)

#> chargement du reseau moleculaire de madbyte
def load_network_madbyte(output_dir,name="madbyte", type="all"):
    G = None
    if type == "all":
        G = read_graphml(
            f"{output_dir}/{name}_association_network_all.graphml")
    if type == "sim":
        try:
            G = read_graphml(
                f"{output_dir}/{name}_similarity_network_network.graphml")
        except:
            pass
    if type == "hybrid":
        try:
            G = read_graphml(f"{output_dir}/{name}_hybrid_network.graphml")
        except:
            pass
    return G

# reconstructions des déplacements du molécules contenus dans ses systèmes de spins
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
    for name in listdir(input_dir):
        spectre = {}
        spectre["name"] = name
        hsqc_toscy = ""
        with open(f'{output_dir}/{name}/{name}_spin_systems.json', 'r') as f:
            for ligne in f:
                hsqc_toscy = f'{hsqc_toscy}{ligne}'
        spectre["hsqc_toscy"] = hsqc_toscy
        spectres.append(spectre)
        
    # construction de dataframe des spectres en regroupant pour chaque molécule, tous les déplacements chimiques
    df_2drmn = DataFrame(spectres)
    #df_2drmn.to_json('{output_dir}/hsqc_tocsy_coupled.json')
    df_2drmn["hsqc_toscy"] = df_2drmn["hsqc_toscy"].apply(loads)
    df_2drmn["hsqc_toscy"] = df_2drmn["hsqc_toscy"].apply(combined_spin_system_for_sample)
    df_2drmn.to_json('hsqc_tocsy_ml.json')
    return df_2drmn
    
# > construction des clusters à partir du réseau moléculaire de madbyte

## calculer les feuilles d'un arbre
def get_leaves(tree, root_node):
    succs = list(tree.successors(root_node))
    if len(succs) == 0 :
        return [root_node]
    leaves = []
    for succ in succs:
        leaves.extend(get_leaves(tree, succ))
    return leaves

## calculer le voisinage d'un noeud depuis
def voisinage_mol_madbyte(x, G, Mols=None, hybrid=False):
    tree = bfs_tree(G, x, depth_limit = 2 if hybrid else 3) 
    return [ node for node in get_leaves(tree, x) if G.nodes[node]["_type"] == "standard" ]


def get_mol_madbyte(G):
    return [ k for k, v in G.nodes(data=True) if v["_type"]=="standard"]