from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# fonction de projection des données en dimension 'd'
def project_data(data, d=2):
    projector = TSNE(n_components=d, random_state=0)
    return projector.fit_transform(data)
    
# fonction de visualisation
def visualize(data2d, labels):
    n = data2d.shape[0]
    data = {}
    
    for i in range(n):
        if labels[i] not in data.keys():
            data[labels[i]] = []
        data[labels[i]].append(i)
    
    for key in data.keys():
        plt.scatter(data2d[data[key], 0], data2d[data[key], 1])
        
    plt.show()

# visualisation d'une dataset
def visualise_dataset(dataset,withtitle=True):
    it = 1
    for x in dataset['X']:
        p = project_data(x.T)
        if withtitle:
            plt.title(f"vue {it}")
        visualize(p, dataset['Y'])
        it += 1