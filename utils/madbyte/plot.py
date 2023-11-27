from bokeh.io import show, output_notebook
from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                          MultiLine, Range1d, ResetTool,)
from bokeh.plotting import figure, from_networkx
from networkx import set_edge_attributes, get_edge_attributes, set_node_attributes, spring_layout



# Modification du réseau pour la visualisation
# "colors_clusters" est un dictionnaire où la clé est le nom de la molécule et la valeur la couleur du clusters
# colors_clusters ( mol : color_cluster )
def modif_madbyte_network(graph , colors_clusters = None):
    # Calcul des atributs de visualisations des aretes
    try:
        max_weight = max(get_edge_attributes(graph, 'weight').values())
    except ValueError:
        max_weight = 1.0
    calc_alpha = lambda x: 0.1 + 0.6 * (x / max_weight)
    edge_attrs = {(s,e): calc_alpha(d['weight']) for s,e,d in graph.edges(data=True)}
    set_edge_attributes(graph, edge_attrs, "_alpha")
    
    # Calcul des atributs de visualisation des noeuds
    node_attrs = {
        k: {
            "_color" : v["_color"] if colors_clusters == None else calc_color(v, colors_clusters, k),
            "_num_members": len(eval(v.get('members', '[]')))
        }
        for k,v in graph.nodes(data=True) }
    set_node_attributes(graph, node_attrs)
    return graph



# fonction pour visualiser le réseau moléculaire de madbyte en ajoutant les couleurs des clusters
def visualize_graph_madbyte(graph, colors_clusters = None):

    # Modification du réseau pour la visualisation
    graph = modif_madbyte_network(graph, colors_clusters=colors_clusters)
    
    #print(graph.nodes(data=True)["HND_Azithromycin"])
    # Créer une figure Bokeh
    plot = figure(title="Interactive Graph", tools="pan,wheel_zoom,box_zoom,reset,save",x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
    
    # Convertir le graphe NetworkX en un graphe Bokeh
    plot_graph = from_networkx(graph, spring_layout, scale=1, center=(0, 0))
    plot_graph.node_renderer.glyph = Circle(size='_size', fill_color="_color")
    plot_graph.node_renderer.selection_glyph = Circle(size=15, fill_color="_color")#works
    plot_graph.node_renderer.hover_glyph = Circle(size=15, fill_color="red")#works
    plot_graph.edge_renderer.glyph = MultiLine(line_alpha="_alpha", line_width=1)
    
    # Ajouter le graphe à la figure
    plot.renderers.append(plot_graph)

    # Ajouter un outil de survol pour afficher les informations des nœuds
    #hover = HoverTool(tooltips=[("Node", "@index"), ("Type", "@_type"), ("Members", "@members"), ("Nombre de membre", "@_num_members")])
    hover = HoverTool(tooltips=[("Node", "@index"), ("Members", "@members")])
    plot.add_tools(hover)

    # Ajouter d'autres outils d'interaction
    plot.add_tools(BoxZoomTool(), ResetTool())

    # Afficher la figure dans le notebook ou dans une fenêtre séparée
    output_notebook()
    show(plot)
    

def calc_color(v, colors_clusters, mol):
    return v["_color"] if v["_type"] == "spin" else colors_clusters[mol]