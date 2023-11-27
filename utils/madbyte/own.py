from os import mkdir, listdir
from networkx import Graph, write_graphml,read_graphml, connected_components
from pandas import read_json, DataFrame
from json import loads
from numpy import zeros, nan
from pathlib import Path

from .madbyte.core import construct_spin_system, construct_correlation_matrix, create_outputs
from .madbyte.utils import trim_associations, hybridize_network, combinations, partition

#> prétraitement et calcul des systèmes de spin de madbyte
def preprocessing_madbyte(input_dir, output_dir):
    try:
        mkdir(f'{output_dir}')
    except FileExistsError:
        pass

    list_samples = []
    for item in listdir(input_dir):
        list_samples.append(item)
        try:
            mkdir(f'{output_dir}/{item}')
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

def association_network(
    project_dir,
    fname,
    corr_mat,
    master,
    colors,
    cutoff=0.5,
    hppm_error=0.05,
    cppm_error=0.5,
    max_system_size=20,
):
    master = master.loc[master['Members'].apply(lambda x: len(x) < max_system_size)]
    idxs = master.Spin_System_ID.tolist()
    systems = [(row.Spin_System_ID, {"members": str(row.Members)}) for row in master.itertuples()]
    extracts = master.Found_In.unique()
    standards, samples = partition(lambda x: x.startswith("HND_"), extracts)
    idx_master = master.set_index("Spin_System_ID", drop=True)

    def get_weight(x,y):
        # Get max value from two ids
        return float(max(corr_mat.loc[x,y], corr_mat.loc[y,x]))

    def same_sample(x,y):
        # tells whether two spin system are from the sample sample
        # based on matching of first 7 char of id
        return idx_master.loc[x, 'Found_In'] == idx_master.loc[y, 'Found_In']

    # Determine edges
    # Edges from extract to systems
    extract_edges = [(row.Spin_System_ID, row.Found_In) for row in master.itertuples()]
    spinsystem_edges = [
        (x,y,get_weight(x,y)) for x,y in combinations(idxs, r=2)
        if get_weight(x, y) >= cutoff and not same_sample(x,y)
    ]

    # network building
    G = Graph()
    G.add_nodes_from(systems, _color=colors['spin'], _type="spin")
    G.add_nodes_from(samples, _color=colors['extract'], _type="extract")
    G.add_nodes_from(standards, _color=colors['standard'], _type="standard")
    G.add_edges_from(extract_edges, weight=1.0)
    G.add_weighted_edges_from(spinsystem_edges)
    write_graphml(G, project_dir.joinpath(f"{fname}_association_network_all.graphml"))
    try: 
        # Filters unconnected
        H = trim_associations(G)
        # write_graphml(G, project_dir.joinpath("test_2.graphml")
        write_graphml(H, project_dir.joinpath(f"{fname}_similarity_network_network.graphml"))
    except: 
        print('Cannot Generate Similarity Network. \n If more than one sample was run, no similarities were found. \n If only was sample was run, please increase the number of samples to compare to in order to generate a similarity network.')
    try: 
        # Join connected associations
        J = hybridize_network(H,idx_master, colors, hppm_error, cppm_error)
        write_graphml(J, project_dir.joinpath(f"{fname}_hybrid_network.graphml"))
    except: 
        print('Cannot Generate Hybrid Network.')
        
def create_outputs(
    project_dir,
    fname="MADByTE",
    threshold=0.5,
    hppm_error=0.05,
    cppm_error=0.5,
    colors=None,
    max_system_size=20,
):
    if not colors:
        colors = {
            "spin": "#009999", # GREY
            "extract": "#ff3333", # RED
            "standard": "#0FFBFF", # Black
        }
    project_dir = Path(project_dir)
    corr_mat = read_json(project_dir.joinpath("correlation_matrix.json"))
    master = read_json(project_dir.joinpath("Spin_Systems_Master.json"), precise_float=True)
    association_network(project_dir, fname, corr_mat, master, colors, cutoff=threshold, hppm_error=hppm_error,
        cppm_error=cppm_error,max_system_size=max_system_size,)


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
def madbyte_clusters_begin(G):
    clusters_mol = {}
    k=0
    for component in connected_components(G):
        for node in component:
            if G.nodes[node]["_type"] == "standard":
                clusters_mol[node] = k
        k = k + 1
    return clusters_mol
    
def madbyte_clusters(G, names_mols):
    # calcul des composantes connexes du réseau qui constituent les clusters de molécules
    clusters_mol = madbyte_clusters_begin
    
    n = len(names_mols)
    mols = zeros(n)
    for i in range(n):
        try:
            mols[i] = clusters_mol[names_mols[i]]
        except :
            mols[i] = nan
            
    return mols