from random import choice
from matplotlib.colors import to_rgb

# calcul de luminosité d'une couleur
def brightness(str_color):
    pr, pg, pb = to_rgb("#D7CE5F")
    return (pr + pg + pb)/3

# generation de "n" coleurs
def generate_random_colors(n, exclude_colors=[], minb=0.3, maxb=0.7):
    # Obtenir la liste des couleurs de base
    i = 0
    colors = []
    while i < n:
        color = "#" + "".join([choice("0123456789ABCDEF") for j in range(6)])
        if color not in exclude_colors:
            b = brightness(color)
            if b > minb and b < maxb :
                colors.append(color)
                i += 1
            
    return colors

# generation de "n" couleurs pour des clusters
def generate_colors_clusters(K, clusters_mol, exclude_colors=[]):
    colors = generate_random_colors(K, exclude_colors=exclude_colors)
    res = { mol : colors[idcluster]  for   mol, idcluster in clusters_mol.items()}
    return res
    
# attache des noms de molécules aux étiquetes de clusters
def attach_names_on_labels(names, labels):
    n = len(names)
    mols = {
        names[i] : labels[i]
        for i in range(n)
    }
    return mols