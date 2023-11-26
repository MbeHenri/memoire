from numpy import zeros, reshape, max
from pandas import merge, DataFrame

# pretraitement des données RMN de molecules
def preprocessing_rmn_data(dataset, min_hppm=2, max_hppm=10,min_cppm=20,max_cppm=150,len_win_hppm=1,len_win_cppm=20,):
    # calcul des dimensions des matrices des spectres
    len_spectre_hppm = int((max_hppm - min_hppm)/len_win_hppm) + 1
    len_spectre_cppm = int((max_cppm - min_cppm)/len_win_cppm) + 1
    
    spectres = dataset["hsqc_toscy"]
    
    # trasformation des déplacements de spectres ( en utilisant des fenetres sur les déplacements hppm et cppm )
    #* un indice de colonne correspond à une plage des déplacements des atomes "H", 
    #* un indice de ligne correspond à une plage des déplacements des atomes "C"
    # Ainsi une molécule devient une matrice 
    len_spectres = spectres.shape[0]
    spectres_for_ml = zeros((len_spectres, len_spectre_cppm, len_spectre_hppm), dtype=float)
    
    for i in range(len_spectres):
        for peak in spectres[i]:
            hppm = peak[0]
            cppm = peak[1]
            if min_hppm <= hppm and hppm <= max_hppm and min_cppm <= cppm and cppm <= max_cppm:
                id_hppm = int((hppm - min_hppm)/len_win_hppm)
                id_cppm = int((cppm - min_cppm)/len_win_cppm)
                spectres_for_ml[i][id_cppm][id_hppm] += 1
                
    # transformation des matrices de molécules en vecteur
    spectres_for_ml = reshape(spectres_for_ml, (len_spectres,
               spectres_for_ml[0].shape[0]*spectres_for_ml[0].shape[1]))
    
    return spectres_for_ml

# pretraitement des donées ms de molécules
def preprocessing_ms_data(dataset, min_mz=10,max_mz=1000,min_intensity=0.01,len_win_mz = 1):
    
    # longeuer d'un vecteur de spectre pour un algorithme d'apprentissage
    len_spectre_mz = int((max_mz - min_mz)/len_win_mz) + 1
    
    spectres = dataset["peaks"]
    
    # trasformation des radicaux de molécules (en utilisant des fenetres de masse/charge)
    # * un indice correspond à une plage des masses sur charge des radicaux de l'exemple
    len_spectres = spectres.shape[0]
    spectres_for_ml = zeros((len_spectres, len_spectre_mz), dtype=float)
    
    for i in range(len_spectres):
        for peak in spectres[i]:
            mz = peak[0]
            intensity = peak[1]
            if min_mz <= mz and mz <= max_mz:
                id = int((mz - min_mz)/len_win_mz)
                spectres_for_ml[i][id] += intensity
    
    # normalisation des intensites des radicaux pour chaque spectre de molécule
    max_intensities = max(spectres_for_ml, axis=1)
    spectres_for_ml = spectres_for_ml / reshape(max_intensities, (max_intensities.shape[0], 1))
    
    return spectres_for_ml

# on join les spectres en fonction de leur nom
# on fait correspondre les spectres de rmn et les spectres 
def joining_data(data_ms, data_rmn):
    
    data =  merge(data_ms, data_rmn, on="names", how="inner")
    dataset_ms, dataset_rmn = DataFrame({"peaks": data["peaks"]}), DataFrame({"hsqc_toscy": data["hsqc_toscy"]})
    return dataset_ms, dataset_rmn, data["names"]

# pretraitement des spectres ms et rmn pour les algorithmes de machin-learning 
# inspiré de MSpectrAI
def preprocessing_for_mc(data_ms, data_rmn):
    
    dataset_ms, dataset_rmn, names = joining_data(data_ms, data_rmn)
    
    spectres_ms = preprocessing_ms_data(dataset_ms)
    spectres_rmn = preprocessing_ms_data(dataset_rmn)
    
    dataset = {"X" : [spectres_ms.T, spectres_rmn.T], "names" : names }

    return dataset