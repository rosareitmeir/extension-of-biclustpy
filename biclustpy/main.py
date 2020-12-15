import grasp
import gvns
import ils
import helpers
import ilp
import ch
import preprocess
import time
import numpy as np

class Algorithm:
    
    """Class to select algorithm for solving the subproblems.
    
    Attributes:
        algorithm_name (string): Name of selected algorithm. 
            Options: \"ILP\", \"CH\". 
            Default: \"ILP\".
        ilp_time_limit (float): Time limit for algorithm \"ILP\" in seconds. 
            If <= 0, no time limit is enforced. 
            Default: 60.
        ilp_tune (bool): If True, the model generated by \"ILP\" is tuned before being optimized. 
            Default: False.
        ch_alpha (float): Between 0 and 1. If smaller than 1, the algorithm behaves non-deterministically.
            Default: 1.0.
        ch_seed (None or int): Seed for random generation. 
            Default: None.
    """
    
    def __init__(self):
        self.algorithm_name = "ILP"
        self.ilp_time_limit = 60
        self.ilp_tune = False
        self.ch_alpha = 1.0
        self.grasp_alpha=0.7
        self.seed = None
        self.max_iter = 10
        self.nmin = 2
        self.nmax = 10
        self.meta_time_limit= np.inf
        self.grasp_time_limit= np.inf
    
    def use_ilp(self, time_limit = 60, tune = False):
        """Use the algorithm \"ILP\".
            
        Args:
            time_limit (float): Time limit for algorithm \"ILP\" in seconds. If <= 0, no time limit is enforced.
            tune (bool): If True, the model generated by \"ILP\" is tuned before being optimized.
        """
        self.algorithm_name = "ILP"
        self.ilp_time_limit = time_limit
        self.ilp_tune = tune
    
    def use_ch(self, alpha = 1.0, seed = None):
        """Use the algorithm \"CH\".
        """
        self.algorithm_name = "CH"
        self.ch_alpha = alpha
        self.seed = seed


    def use_GRASP(self,max_iter=10,alpha=0.7,seed=None,time_limit= np.inf):  # Greedy Randomized Adaptive Search Procedure
        """Use the algorithm \"GRASP\".
                """
        self.algorithm_name="GRASP"
        self.grasp_alpha=alpha
        self.max_iter=max_iter
        self.seed = seed
        self.grasp_time_limit=time_limit
    # metaheuristics

    def use_GVNS(self,max_iter=20,nmin=2,nmax=10, time_limit= np.inf):  # General Variable Neighborhood Search (GVNS)
        """Use the algorithm \"GVNS\".
                """
        self.algorithm_name="GVNS"
        self.max_iter = max_iter
        self.nmin = nmin
        self.nmax = nmax
        self.meta_time_limit = time_limit

    def use_ILS(self,max_iter=20,nmin=2,nmax=10, time_limit= np.inf):  # Iterated Local Search
        """Use the algorithm \"ILS\".
                """
        self.algorithm_name = "ILS"
        self.max_iter=max_iter
        self.nmin=nmin
        self.nmax=nmax



            
    def run(self, weights, subgraph, obj_val=None):
        """Runs the selected algorithm on a given subgraph.
        
        Args:
            weights (numpy.array): The overall problem instance.
            subgraph (networkx.Graph): The subgraph of the instance that should be rendered bi-transitive.
        
        Returns:
            networkx.Graph: The obtained bi-transitive subgraph.
            float: Objective value of obtained solution.
            bool: True if and only if obtained solution is guaranteed to be optimal.
        """
        if self.algorithm_name == "ILP":
            return ilp.run(weights, subgraph, self.ilp_time_limit, self.ilp_tune)
        elif self.algorithm_name == "CH":
            return ch.run(weights, subgraph, self.ch_alpha, self.seed)
        elif self.algorithm_name == "GRASP":
            return grasp.run(weights, subgraph, self.max_iter, self.grasp_alpha, self.seed, self.grasp_time_limit)
        # metaheuristics
        elif self.algorithm_name == "GVNS":
            return gvns.run(weights, subgraph,obj_val, self.max_iter,self.nmin,self.nmax, self.meta_time_limit)
        elif self.algorithm_name == "ILS":
            return ils.run(weights, subgraph,obj_val,self.max_iter,self.nmin,self.nmax, self.meta_time_limit)
        else:
            raise Exception("Invalid algorithm name \"" + self.algorithm_name + "\". Options: \"ILP\", \"CH\",\"GRASP\",\"ILS\",\"GVNS\" .")
    
    
def compute_bi_clusters(weights, preprocessing_method, algorithm, metaheurisitc=None ):
    """Computes bi-clusters using bi-cluster editing.
    
    Given a matrix W = (w[i][k]) of weights of dimension n x m with positive and negative 
    entries, the bi-cluster editing problem asks to transform the bipartite graph
    ([n], [m], E) into a collection of disjoint bi-cliques by adding or deleting edges:
        - The edge set E contains all (i,k) with w[i][k] > 0.
        - Adding an edge induces the cost -w[i][k].
        - Deleting an edge induces the cost w[i][k].
        - The overall induced cost should be minimized.
    
    The function first decomposes the instance into connected components and 
    checks whether they are already bi-cliques. Subsequently, it calls a 
    user-specified algorithm to solve the remaining subproblems.
    
    Args:
        weights (numpy.array): The problem instance.
        algorithm (Algorithm): The subgraph that should be rendered bi-transitive.
        metaheuristic (Algorithm) : improves initial solution, which is created by algorithm
    
    Returns:
        list of tuple of list of int: List of computed bi-clusters. 
            The first element of each bi-cluster is the list of rows, the second the list of columns.
        float: Objective value of the obtained solution.
        bool: True if and only if the obtained solution is guaranteed to be optimal.
    """
    # measure time
    start_time= time.time()

    
    # Get dimension of the problem instance and build NetworkX graph.
    num_rows = weights.shape[0]
    num_cols = weights.shape[1]
    graph = helpers.build_graph_from_weights(weights, range(num_rows + num_cols))
    
    # Initialize the return variable.
    bi_clusters = []
    # save removed nodes from Rule 2 or New Rule to add them to final solution
    removed_nodes={}
    # Decompose graph into connected components and check if some 
    # of them are already bi-cliques. If so, put their rows and columns 
    # into bi-clusters. Otherwise, add the connected 
    # component to the list of connected subgraphs that have to be 
    # rendered bi-transitive.
    subgraphs = []
    components = helpers.connected_components(graph)
    for component in components:
        if helpers.is_bi_clique(component, num_rows):
            bi_cluster = ([], [])
            for node in component.nodes:
                if helpers.is_row(node, num_rows):
                    bi_cluster[0].append(node)
                else:
                    bi_cluster[1].append(node)
            bi_clusters.append(bi_cluster)
        else:
            # Rule 2 or New Rule
            if preprocessing_method == "Rule 2":
                removed_nodes.update(preprocess.execute_Rule2(component))
            else:
                removed_nodes.update(preprocess.execute_NewRule(component,weights,num_rows))
            subgraphs.append(component)

    # Print information about connected components.
    print("\n==============================================================================")
    print("Finished pre-processing with "+ preprocessing_method+ ".")
    print("------------------------------------------------------------------------------")
    print("Number of connected components: " + str(len(components)))
    print("Number of bi-cliques: " + str(len(bi_clusters)))
    print("Number of removed nodes: "+ str(sum([len(v)for v in removed_nodes.values()])))
    print("==============================================================================")
    
    # Solve the subproblems and construct the final bi-clusters. 
    # Also compute the objective value and a flag that indicates whether the
    # obtained solution is guaranteed to be optimal.
    obj_val = 0
    is_optimal = True 
    counter = 0
    for subgraph in subgraphs:
        counter = counter + 1
        print("\n==============================================================================")
        print("Solving subproblem " + str(counter) + " of " + str(len(subgraphs)) + ".")
        print("------------------------------------------------------------------------------")
        n = len([node for node in subgraph.nodes if helpers.is_row(node, num_rows)])
        m = len([node for node in subgraph.nodes if helpers.is_col(node, num_rows)])
        print("Dimension: " + str(n) + " x " + str(m))
        bi_transitive_subgraph, local_obj_val, local_is_optimal = algorithm.run(weights, subgraph)

        # improve solution by chosen metaheuristic: GVNS or ILS
        # returns improved graph and its objective value 
        if metaheurisitc != None:
            print("Optimizing constructed bi-transitive subgraph with "+ metaheurisitc.algorithm_name+".")
            bi_transitive_subgraph, local_obj_val, local_is_optimal = metaheurisitc.run(weights, bi_transitive_subgraph, local_obj_val)

        obj_val = obj_val + local_obj_val
        is_optimal = is_optimal and local_is_optimal
        for component in helpers.connected_components(bi_transitive_subgraph):
            if not helpers.is_bi_clique(component, num_rows):
                msg = "Subgraph should be bi-clique but isn't."
                msg = msg + "\nNodes: " + str(component.nodes)
                msg = msg + "\nEdges: " + str(component.edges)
                raise Exception(msg)
            bi_cluster = ([], [])
            for node in component.nodes:
                if helpers.is_row(node, num_rows):
                    bi_cluster[0].append(node)
                    # adding removed nodes from New Rule or Rule 2
                    if node in removed_nodes:
                        bi_cluster[0].extend(removed_nodes[node])
                else:
                    bi_cluster[1].append(node)
                    # adding removed nodes from New Rule or Rule 2
                    if node in removed_nodes:
                        bi_cluster[1].extend(removed_nodes[node])



            bi_clusters.append(bi_cluster)
        print("==============================================================================")

    execution_time= time.time()-start_time
    print("\n==============================================================================")
    print("Finished computation of bi-clusters.")
    print("------------------------------------------------------------------------------")
    print("Objective value: " + str(obj_val))
    print("Is optimal: " + str(is_optimal))
    print("Number of bi-clusters: " + str(len(bi_clusters)))
    print("==============================================================================")
    
    
    # Return the obtained bi-transitive subgraph, the objective value of the obtained solution, 
    # and a flag that indicates if the solution is guaranteed to be optimal.
    return bi_clusters, obj_val, is_optimal , execution_time
    
def save_bi_clusters_as_xml(filename, bi_clusters, obj_val, is_optimal, time, instance = "", names=None):
    """Saves bi-clusters as XML file.
    
    Args:
        filename (string): Name of XML file.
        bi_clusters (list of tuple of list of int): List of computed bi-clusters.
            The first element of each bi-cluster is the list of rows, the second the list of columns.
        obj_val (float): Objective value of the obtained solution.
        is_optimal (bool): Set to True if and only if the obtained solution is guaranteed to be optimal.
        instance (string): String that contains information about the problem instance.
    """
    elem_tree = helpers.build_element_tree(bi_clusters, obj_val, is_optimal,time, instance, names)
    xml_file = open(filename, "w")
    xml_file.write(helpers.prettify(elem_tree))
    xml_file.close()
    
