from utils.gnps.own import load_data_gnps, gnps
from utils.madbyte.own import preprocessing_madbyte, madbyte, load_network_madbyte, load_rmn_data_for_ml
from utils.clusters_network import gnps_clusters, madbyte_clusters
from utils.preprocessingml import preprocessing_for_mc

def process_madbyte_gnps(pathdirRMN ="rmn", pathdirMS="ms", output_dir="temp"):
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
    labels_madbyte = madbyte_clusters(network_madbyte_hybrid, dataforml["names"], hybrid=True)
    
    params_gnps = None
    if labels_gnps != None:
        params_gnps = {"labels": labels_gnps}
        
    params_madbyte = None
    if labels_madbyte != None:
        params_madbyte = {"labels": labels_madbyte}
        
    return network_gnps, networks_madbyte, dataforml, params_gnps, params_madbyte 
