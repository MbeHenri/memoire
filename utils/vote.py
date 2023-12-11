from utils.ClusterEnsembles.ClusterEnsembles import mcla

#
def vote(labels_weaks, k, weights_weaks=None):
    return mcla(labels_weaks, k, 0)
    