from scipy.io import loadmat
from numpy import array, reshape, concatenate, loadtxt


def load_msrcv1(path_dir_datasets=""):

    dataset = loadmat(f'{path_dir_datasets}msrc-v1/MSRC-v1.mat')
    X = []
    for x in dataset["X"][0]:
        X.append(array(x.T, dtype=float))
    dataset = {"X": X, "Y": reshape(dataset["Y"], (dataset["Y"].shape[0]))}
    dataset["name"] = "MSRC-v1"

    return dataset


def load_multiple_features(path_dir_datasets=""):

    fourier_view = loadtxt(f"{path_dir_datasets}multiple+features/mfeat-fou")
    f_corr_view = loadtxt(f"{path_dir_datasets}multiple+features/mfeat-fac")
    kar_view = loadtxt(f"{path_dir_datasets}multiple+features/mfeat-kar")
    pix_avg_view = loadtxt(f"{path_dir_datasets}multiple+features/mfeat-pix")
    zer_moment = loadtxt(f"{path_dir_datasets}multiple+features/mfeat-zer")
    morphological_view = loadtxt(
        f"{path_dir_datasets}multiple+features/mfeat-mor")

    X = [fourier_view.T, f_corr_view.T, kar_view.T,
         pix_avg_view.T, zer_moment.T, morphological_view.T]
    Y = concatenate(array([[i for _ in range(200)]
                    for i in range(10)]), axis=0)

    dataset = {"X": X, "Y": Y}
    dataset["name"] = "multiple_features"

    return dataset


def load_bbcsport(path_dir_datasets=""):

    dataset = loadmat(f'{path_dir_datasets}bbcsport/BBCSport.mat')
    X = []
    for x in dataset["X"][0]:
        X.append(x.toarray())
    dataset = {"X": X, "Y": reshape(dataset["gt"], (dataset["gt"].shape[0]))}
    dataset["name"] = "BBCSport"

    return dataset
