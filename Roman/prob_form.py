import numpy as np
import pandas as pd
import scipy as sp
from scipy.spatial.kdtree import distance_matrix
import matplotlib.pyplot as plt
from termcolor import colored
from gurobipy import Model,GRB,LinExpr


class problem_formulation:

    def __init__(self, path,n_vehicles,cap_vehicle,min_load):

        # The initialization step does the following:
            # - Load the data
            # - Separate it into a user specified ammount fo clusters
            # (We can also change it to read a predefined list of clusters)
            # Generate a matrix containing all the relevant information self.node_info
            # Create the decision variables
            # Display data 
        
        # Load data and separate into clusters
        self.node_info = pd.read_excel(path).to_numpy().T

        # Find weights of coneccting arcs
        from scipy.spatial import distance_matrix
        self.weights_of_arcs = distance_matrix(self.node_info.T[:,1:3], self.node_info.T[:,1:3])
        #Change shape so it matches the decision variables (Drop the 0s)
        self.weights_of_arcs = self.weights_of_arcs[self.weights_of_arcs != 0].reshape((40,39))
        

        #Other Parameters
        self.n_vehicles = n_vehicles
        self.cap_vehicle = cap_vehicle
        self.min_load = min_load

        # Initialize model
        self.model = Model()


        # Create decision variables
        self.decision_vars = []
        #Add the decision variables to the model
        x,y = np.shape(self.weights_of_arcs)
        for i in range(x):
            for j in range(y):
                    if i < j:
                        self.decision_vars.append(self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY,name="x[%s,%s]"%(i+1,j+1)))
                    else:
                        self.decision_vars.append(self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY,name="x[%s,%s]"%(i+1,j+2)))

        self.decision_vars = [self.decision_vars[i:i+39] for i in range(0, len(self.decision_vars), 39)]
        
        self.model.update()

        print(colored('Successfully loaded data and initialized model', 'green'))
        print(len(self.decision_vars))

    def cluster(self, N):

        # Cluster number variable 
        self.N = N
        #Generate Clusters (You can iterate till you have a clustering you like)
        from sklearn.cluster import KMeans
        kmeans = KMeans(init='k-means++', n_clusters=N, n_init=10)
        kmeans.fit(self.node_info[[1,2],:].T)
        cluster_data = kmeans.predict(self.node_info[[1,2],:].T)
        self.node_info = np.vstack((self.node_info, cluster_data))
        print(colored('Sucessfully created clusters', 'green'))


    def demand (self,n=None,random=False,u=None,l=None):
       
        if random:
            demand = np.random.randint(low=l,high=u+1, size =len(self.node_info.T))
        else:
            demand = np.ones(len(self.node_info.T))*n
        
        self.node_info = np.vstack((self.node_info, demand))
        print(colored('Sucessfully assigned demands', 'green'))
    

    def add_origin_depot(self,x,y):

        self.origin = [x,y]
        
        # Find Costs
        origin_weights = [np.linalg.norm([self.node_info[1][n]-x,self.node_info[2][n]-y]) for n in range(len(self.node_info.T))]

        origin_vars = []     

        for n in range(len(self.node_info.T)):
            origin_vars.append(self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY,name="x[%s,%s]"%(0,n+1)))

        # Append to the weights of arcs matrix
        transpose = np.array([origin_weights])
        self.weights_of_arcs = np.hstack((self.weights_of_arcs, transpose.T))
        self.weights_of_arcs = np.vstack((self.weights_of_arcs, np.array([origin_weights])))
        
        # Append to the decision variables

        for n in range(len(self.decision_vars)):
            
            self.decision_vars[n].append(self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY,name="x[%s,%s]"%(n+1,0)))
        
        self.decision_vars.append(origin_vars)

        self.model.update()

        print(colored('Sucessfully added origin depot and created relevant decision variables', 'green'))


    def display_results(self, title, final=False):


        # Remember to change it to work with the home depot

        # Create Plot
        fig, ax = plt.subplots()
        ax.scatter(self.node_info[1], self.node_info[2], c = self.node_info[3], cmap = 'hsv', marker='.',s=50)
        ax.scatter(self.node_info[1], self.node_info[2], s = 7500, alpha=0.2, c = self.node_info[3], cmap = 'hsv', linewidth=0, marker='o')
        fig.suptitle(title, fontsize = 24)

        #Annotate plot
        for n, txt in enumerate(self.node_info[0]):
            ax.annotate(txt, (self.node_info[1][n], self.node_info[2][n]), fontsize=7)

        if final:
            # Draw Connections
            x,y = np.shape(self.weights_of_arcs)

            for i in range(x):
                for j in range(y):
                    if self.decision_vars[i][j]:

                        if i < j:

                            x1 = self.node_info[1][i]
                            x2 = self.node_info[1][j]
                            y1 = self.node_info[2][i]
                            y2 = self.node_info[2][j]
                            
                            
                        else:
                            x1 = self.node_info[1][i]
                            x2 = self.node_info[1][j+1]
                            y1 = self.node_info[2][i]
                            y2 = self.node_info[2][j+1]
                        
                        ax.plot([x1,x2], [y1,y2], color = 'black', linewidth = 2)
                        

        plt.show()


    def degree_constraints(self):
        
        # Variable holding data on clusters
        clusters = self.node_info[3]

        #Single outgoing arc per cluster
        for n in range(self.N):

            # Find indexes of cluster for condition
            indexes = np.where(clusters == n)[0]
            
            
            #Variables of interest

            # Select only connections from variables withing the cluster (First sigma condition)
            voi = [self.decision_vars[i] for i in indexes]
            
            const = LinExpr()

            # Remove connections withing the same cluster (second sigma condition)
            for n, v in enumerate(voi):

                for i in indexes:

                    if i < indexes[n]:
                        del v[i]
                    elif i == indexes[n]:
                        pass
                    else:
                        del v[i-1]
                    
                print(len(v))
                for x in v:
                    const += x 



            

















problem = problem_formulation('dataproblem1.xlsx',5,20,10)

problem.cluster(10)
problem.demand(random=True, u=5, l=0)
problem.add_origin_depot(0,0)
problem.degree_constraints()
#problem.display_results('dghkdvs')