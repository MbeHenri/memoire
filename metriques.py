
from numpy import trace, sum, max, array
from sklearn.metrics import normalized_mutual_info_score # information mutuelle normalisée
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

# exactitude
def accuracy_clustering(labels, labels_clusters):
    cm = confusion_matrix(labels, labels_clusters)
    # on trouve l'allocation idéale en resolvant le problème de maximisation transformé en problème de minimisation
    _, ColCorrect = linear_sum_assignment(-array(cm))
    cm = cm[:, ColCorrect]
    return trace(cm) / sum(cm)

# pureté
def purity(labels, labels_clusters):
    cm = confusion_matrix(labels, labels_clusters)
    return sum(max(cm, axis=0)) / sum(cm)


def clusteringMeasure(labels, labels_clusters):
    return {
        "ACC": accuracy_clustering(labels, labels_clusters),
        "NMI": normalized_mutual_info_score(labels, labels_clusters),
        "PUR": purity(labels, labels_clusters),
    }
