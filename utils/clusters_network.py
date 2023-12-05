
from networkx import connected_components
from .utils import finalise_clusters
from .madbyte.own import get_mol_madbyte, voisinage_mol_madbyte


def voisinage_default(x, G, Mols=None):
    return [x for x in G.neighbors(x)]


def getnode_default(G):
    return [x for x in G.nodes()]

# MOLDBSCAN


def moldbscan(G, Minpts=2, voisinage=voisinage_default, getnode=getnode_default):
    # On voit les clusters comme des zones à forte densité

    # initialisation de clusters
    clusters = {}
    k = -1

    # initialisation des variables utiles
    X = Mols = getnode(G)
    NoisePoint = []
    Visite = {x: False for x in X}

    while X != []:
        x = X.pop()
        Visite[x] = True
        N = voisinage(x, G, Mols=Mols)

        if len(N) >= Minpts:
            # on construit un nouveau cluster
            k += 1
            clusters[x] = k

            # on recupere les voisins non visités et les rend visités, en
            # les ajoutant dans le nouveau cluster
            A = [v for v in N if not Visite[v]]
            for v in A:
                Visite[v] = True
                clusters[v] = k
                X.remove(v)

            # on étend le cluster
            STACK = A
            while STACK != []:
                x_ = STACK.pop()
                N = voisinage(x_, G, Mols=Mols)

                if len(N) >= Minpts:
                    # marquage des points de A comme visité et ajout dans le cluster
                    A = [v for v in N if not Visite[v]]
                    for v in A:
                        Visite[v] = True
                        clusters[v] = k
                        X.remove(v)

                    # on continue détendre le cluster
                    STACK.extend(A)
        else:
            NoisePoint.append(x)

    return clusters, NoisePoint, k+1

# MADbyTE


def madbyte_clusters_components(G):
    # On voit les clusters comme des composentes connexes
    clusters_mol = {}
    k = 0
    for component in connected_components(G):
        for node in component:
            if G.nodes[node]["_type"] == "standard":
                clusters_mol[node] = k
        k = k + 1
    return clusters_mol


def madbyte_clusters_base(G, MinPts=2, hybrid=False):

    def voisinage(x, G, Mols=None):
        return voisinage_mol_madbyte(x, G, Mols=Mols, hybrid=hybrid)

    #[G_sim.subgraph(c).copy() for c in nx.connected_components(G_sim)]
    clusters, noises, _ = moldbscan(
        G, Minpts=MinPts, voisinage=voisinage, getnode=get_mol_madbyte)
    # clusters, noises = madbyte_clusters_components(G), []
    return clusters, noises


def madbyte_clusters(G, names_mols, hybrid=False, MinPts=2):
    clusters, _, _ = madbyte_clusters_base(G, MinPts=MinPts, hybrid=hybrid)
    return finalise_clusters(G, clusters, names_mols)

# GNPS


def gnps_clusters_base(G, MinPts=2):
    clusters, noises, _ = moldbscan(G, Minpts=MinPts)
    return clusters, noises


def gnps_clusters(G, names_mols, MinPts=2):
    clusters, _ = gnps_clusters_base(G, MinPts=MinPts)
    return finalise_clusters(G, clusters, names_mols)
