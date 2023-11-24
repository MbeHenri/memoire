from utils.ClusterEnsembles.ClusterEnsembles import mcla

#
def vote(labels_weaks, weights_weaks, k):
    return mcla(labels_weaks, k, 0)
    