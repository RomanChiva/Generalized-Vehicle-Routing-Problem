from platform import node
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster



def import_data(path, N):
    #:oad data from excel sheet
    excel_data = pd.read_excel(path).to_numpy().T
    print(np.shape(excel_data))

    #Find Clusters
    from sklearn.cluster import KMeans
    kmeans = KMeans(init='k-means++', n_clusters=N, n_init=10)
    kmeans.fit(excel_data[[1,2],:].T)
    cluster_data = kmeans.predict(excel_data[[1,2],:].T)
    print(np.shape(cluster_data))
    node_info = np.vstack((excel_data, cluster_data))

    return node_info

node_info = import_data('dataproblem1.xlsx',10)




def decision_vars(n):
    return np.zeros([n, n-1])



def display_results(node_info, decision_vars):

    # Create Plot
    fig, ax = plt.subplots()
    ax.scatter(node_info[1], node_info[2], c = node_info[3], cmap = 'hsv', marker='.',s=50)
    ax.scatter(node_info[1], node_info[2], s = 7500, alpha=0.2, c = node_info[3], cmap = 'hsv', linewidth=0, marker='o')
    fig.suptitle('Generalized Vehicle Routing Problem', fontsize = 24)

    #Annotate plot
    for n, txt in enumerate(node_info[0]):
        ax.annotate(txt, (node_info[1][n], node_info[2][n]), fontsize=7)


    # Draw Connections
    x,y = np.shape(decision_vars)

    for i in range(x):
        for j in range(y):
            if decision_vars[i][j]:

                if i < j:

                    x1 = node_info[1][i]
                    x2 = node_info[1][j]
                    y1 = node_info[2][i]
                    y2 = node_info[2][j]
                    
                    
                else:
                    x1 = node_info[1][i]
                    x2 = node_info[1][j+1]
                    y1 = node_info[2][i]
                    y2 = node_info[2][j+1]
                
                ax.plot([x1,x2], [y1,y2], color = 'black', linewidth = 2)
                    

    plt.show()


def weights_of_arcs(node_info):
    
    from scipy.spatial import distance_matrix

    weights = distance_matrix(node_info.T[:,1:3], node_info.T[:,1:3])

    return weights














#### GRAVEYARD #####

# Shade Cluster Regions
'''
    # Convex hull generates the boundaries of the region
    from scipy.spatial import ConvexHull

    points = node_info.T[:, 1:3]
    for i in np.unique(node_info[3]):
        in_cluster = [node_info[3] == i]
        points_in_cluster = points[in_cluster]

        if len(points_in_cluster)==1:
            circle = plt.Circle((points_in_cluster[0,0],points_in_cluster[0,1]), alpha = 0.3, radius=5)
            ax.add_artist(circle)
        if len(points_in_cluster)==2:
            ax.plot(points_in_cluster[:,0], points_in_cluster[:,1], alpha = 0.3, linewidth=30)

        if len(points_in_cluster)>2:

            # Create Hull
            hull = ConvexHull(points_in_cluster)

            x_hull = np.append(points_in_cluster[hull.vertices,0],
                        points_in_cluster[hull.vertices,0][0])
            y_hull = np.append(points_in_cluster[hull.vertices,1],
                        points_in_cluster[hull.vertices,1][0])
            
            ax.fill(x_hull, y_hull, alpha=0.3, )



I've been a music enthusiast as far back as I can remember. For a long time I've been looking for ways to combine my passion for music with my technical abilities and I find this internship might just be the right opportunity to do so.

'''
