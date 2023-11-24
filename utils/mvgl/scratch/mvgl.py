from scipy.sparse.csgraph import connected_components
from .learningS import learningS
from .learningA import learningA
from .updateIndicator import updateIndicator

def mvgl(X, k, beta=1, gamma=1, hand_first_k_neighbour=True, hand_graph_neighbour=True, epsilon=1e-3, verbose=False):

    # > appretissage des graphes de similarité pour chacune des vues
    S = learningS(X, k, beta=beta)

    # > fusion des graphes de similarités
    objectives, A = learningA(S, k, epsilon=epsilon, verbose=verbose, gamma=gamma)
    
    # > attribution des observations aux clusters
    _,labels=  connected_components((A+A.T)/2)

    _, P = updateIndicator(A, k)
    return {"A": A, "P": P.real, "obj": objectives, "labels": labels}
