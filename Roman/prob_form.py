from lib2to3.pgen2.token import GREATEREQUAL
import numpy as np
import pandas as pd
import scipy as sp
from scipy.spatial.kdtree import distance_matrix
import matplotlib.pyplot as plt
from termcolor import colored
from gurobipy import Model,GRB,LinExpr


class problem_formulation:

    def __init__(self, path,n_vehicles,cap_vehicle,min_load):

           
        # Load data and separate into clusters
        
        self.node_info = pd.read_excel(path)[:10].to_numpy().T
        print(self.node_info)
        
        #Other Parameters
        self.n_vehicles = n_vehicles
        self.cap_vehicle = cap_vehicle
        self.min_load = min_load

        # Initialize model
        self.model = Model()

        print(colored('Successfully loaded data and initialized model', 'green'))

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

        print(colored('Sucessfully added origin depot', 'green'))

    def decision_variables(self):

        # Create a dictionary containing all decision variables
        self.decision = {}

        # List containing the names of all the nodes (including home node)
        self.IDs = [0] + [int(n) for n in self.node_info[0]]
        
        # Loop to create the decsion variables and add them to the dictionary
        for node in self.IDs:
            #Remove ID of current node so there are no variables for travelling to the same node
            other_IDs = [x for x in self.IDs if x != node]

            for i in other_IDs:
                #Define the variable
                self.decision['%s,%s'%(node,i)] = self.model.addVar(vtype=GRB.BINARY,name='x[%s,%s]'%(node,i))



        # Find the costs
        self.costs = {}

        # Array containing all points starting from the origin depot
        locations = np.vstack((self.origin, self.node_info[1:3].T))

        for node in self.IDs:
            #Remove ID of current node so there are no variables for travelling to the same node
            other_IDs = [x for x in self.IDs if x != node]

            for i in other_IDs:
                # Caluculate the cost
                self.costs['%s,%s'%(node,i)] = np.linalg.norm(locations[i]-locations[node])
        
        
        self.model.update()
        print(colored('Sucessfully created decision variables and clculated associated costs', 'green'))


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
        
        # Variable holding data on clusters (First one is the 0 node whih doesnt beling to any cluster)
        clusters = np.hstack((np.array([-1]),self.node_info[3]))
        
        #Single outgoing/incoming arc per cluster ------------------------------------------------------


        for n in range(self.N): 

            # Find IDs of all nodes in cluster
            nodes_in_cluster = [n for n in np.where(clusters == n)[0]]
            
            # Find IDs of all other nodes where those nodes link
            other_nodes = [n for n in self.IDs if n not in nodes_in_cluster][1:]
            
            # Create the linear expressions
            
            lhs_out = LinExpr()
            lhs_in = LinExpr()

            for i in nodes_in_cluster:
                for j in other_nodes:

                    lhs_out += self.decision['%s,%s'%(i,j)]
                    lhs_in += self.decision['%s,%s'%(j,i)]
            
            # Add COnstraints to the model
            self.model.addConstr(lhs=lhs_out, sense = GRB.EQUAL, rhs=1, name = 'DegreeConstraintOutgoing:%s'%(n))
            self.model.addConstr(lhs=lhs_in, sense = GRB.EQUAL, rhs=1, name = 'DegreeConstraintIncoming:%s'%(n))

        # Constraints for home depot---------------------------------------------------------

        # Create the linear expressions
        lhs_out_home = LinExpr()
        lhs_in_home = LinExpr()
        
        for n in self.IDs[1:]:
            lhs_out_home += self.decision['%s,%s'%(0,n)]
            lhs_in_home += self.decision['%s,%s'%(n,0)]
        
        # Add COnstraints to the model
        self.model.addConstr(lhs=lhs_out_home, sense = GRB.EQUAL, rhs=self.n_vehicles, name = 'HomeOutgoing')
        self.model.addConstr(lhs=lhs_in_home, sense = GRB.EQUAL, rhs=self.n_vehicles, name = 'HomeIncoming')


        self.model.update()

        print(colored('Sucessfully added degree constraints to the model', 'green'))

    

    def flow_constraints(self):

         # Variable holding data on clusters (First one is the 0 node whih doesnt beling to any cluster)
        clusters = np.hstack((np.array([-1]),self.node_info[3]))
        
        #If an arc comes into a node and arc must also come out of that node ------------------------------------------------------

        for n in range(self.N): 

            # Find IDs of all nodes in cluster
            nodes_in_cluster = [n for n in np.where(clusters == n)[0]]
            
            # Find IDs of all other nodes where those nodes link
            other_nodes = [n for n in self.IDs if n not in nodes_in_cluster][1:]

            # Create the linear expressions
            

            for i in nodes_in_cluster:

                lhs = LinExpr()
                rhs= LinExpr()

                for j in other_nodes:

                    lhs += self.decision['%s,%s'%(i,j)]
                    rhs += self.decision['%s,%s'%(j,i)]
                
                self.model.addConstr(lhs=lhs, sense = GRB.EQUAL, rhs=rhs, name = 'FlowConservationForNode:%s'%(i))


        # Define flows between clusters-------------------------------------------------------------


        # Dictionary to store (y = 1 if clusters are connected, 0 if they are not connected)
        self.y_clust = {}
        list_of_clusters = [-1] + [x for x in range(self.N)]
        
        for p in list_of_clusters:
            
            #List containing all other clusters except p
            other_clusters = [n for n in list_of_clusters if n != p]

            for l in other_clusters:

                # List of nodes in cluster p
                cluster_p = [n for n in np.where(clusters == p)[0]]
                
                # List of nodes in cluster l  
                cluster_l = [n for n in np.where(clusters == l)[0]]

                # Sum of all the vars
                rhs = LinExpr()

                for i in cluster_p:
                    for j in cluster_l:

                        rhs += self.decision['%s,%s'%(i,j)]
                
                self.y_clust['%s,%s'%(p,l)] = self.model.addVar(vtype=GRB.INTEGER,name='y[%s,%s]'%(p,l))

                self.model.addConstr(lhs = self.y_clust['%s,%s'%(p,l)], sense = GRB.EQUAL, rhs=rhs, name='ClustersConnected:{a}to{b}'.format(a=p,b=l))

        self.model.update()

        print(colored('Sucessfully added flow constraints to the model', 'green'))
                        

    def side_and_subtour_elimination(self):

        
        # ======PREP====================


        # Find the demand per cluster
        cluster_demand = []
    
        for n in range(self.N):

            demand = np.sum(self.node_info[4][self.node_info[3]==n])
            cluster_demand.append(demand)
        
        # Minimum demand from other clusters (q bar)

        min_other = []

        for x in range(self.N):

            comp_min = min(cluster_demand[:x] + cluster_demand[x+1:])
            min_other.append(comp_min)

    # Define U (Sketched out by this atm) (U is the vehice load just after leaving a cluster)

        u = self.model.addVars(self.N, vtype=GRB.INTEGER, name='u')

        #================= Constraints=================

        # Ensure vehicle has enough load to deliver at cluster P

        for p in range(self.N): 
            # Left hand side
            lhs = LinExpr()
            lhs += u[p] + (self.cap_vehicle-min_other[p]-cluster_demand[p])*self.y_clust['-1,%s'%p]- min_other[p]*self.y_clust['%s,-1'%p]

            # Right hand side of constraint
            rhs = LinExpr()
            rhs += self.cap_vehicle-min_other[p]

            self.model.addConstr(lhs=lhs, sense= GRB.LESS_EQUAL, rhs=rhs, name='LoadConstraintCluster%s'%p)

        # Comply with minimum load before returning to origin depo

        for p in range(self.N): 
            # Left hand side
            lhs = LinExpr()
            lhs += u[p] + min_other[p]*self.y_clust['-1,%s'%p] +(min_other[p]+cluster_demand[p]- self.min_load)*self.y_clust['%s,-1'%p]

            # Right hand side of constraint
            rhs = LinExpr()
            rhs += min_other[p]+cluster_demand[p]

            self.model.addConstr(lhs=lhs, sense= GRB.GREATER_EQUAL, rhs=rhs, name='MinLoadDeliveredBeforeReturntoDepo%s'%p)

        # No Single Customer Visit TRips

        for p in range(self.N): 

            # Left hand side
            lhs = LinExpr()
            lhs += self.y_clust['-1,%s'%p] + self.y_clust['%s,-1'%p]

            self.model.addConstr(lhs=lhs, sense= GRB.LESS_EQUAL, rhs=1, name='NoSingleCustomerVisits%s'%p)
        

        # Continuity

        for p in range(self.N):
            for l in range(self.N):

                if p!=l:

                    # Left hand side
                    lhs = LinExpr()
                    lhs += u[p] - u[l] + self.cap_vehicle*self.y_clust['{a},{b}'.format(a=p, b=l)] + (self.cap_vehicle - cluster_demand[p] - cluster_demand[l])*self.y_clust['{a},{b}'.format(a=l, b=p)]
                    
                    # Right hand side of constraint
                    rhs = LinExpr()
                    rhs += self.cap_vehicle - cluster_demand[l]

                    self.model.addConstr(lhs=lhs, sense= GRB.LESS_EQUAL, rhs=rhs, name='ContinuityBetweenClusters[%s,%s]'%(p,l))

        
        # Non-Negativity for u
        
        for i in range(len(u)):
            self.model.addConstr(lhs=u[i], sense= GRB.GREATER_EQUAL, rhs=0, name='Non_negativeU[%s]'%i)
        
        for key in self.y_clust:
            self.model.addConstr(lhs=self.y_clust[key], sense= GRB.GREATER_EQUAL, rhs=0, name='Non_negativeY')

        self.model.update()

        print(colored('Sucessfully added Side and Subtour elimination constraints', 'green'))
    

    # Create Objective function
    def add_obj(self):

        self.obj = LinExpr()

        for i in self.IDs:
            for j in self.IDs:

                if i!=j:
                    self.obj += self.decision['{a},{b}'.format(a=i,b=j)]*self.costs['{a},{b}'.format(a=i,b=j)]
        

        self.model.setObjective(self.obj, GRB.MINIMIZE)

        self.model.update()
        
        print(colored('Sucessfully created objective function', 'green'))


    def optimize_model(self):
        self.model.optimize()
        self.model.write('model.lp')

        self.model.computeIIS()
        self.model.write('resullts_trimmed.lp')


