import networkx as nx

import movement


def run(weights, bi_transitive_subgrpah, cur_val):
    edit_matrix= movement.Solution(weights, bi_transitive_subgrpah)
    # while stop conditon ist not met
        #k=1
        #while k <3:
            # shake function ( matrix obj  with current solution, cur_val) creates new random solution saved in matrix obj
            # movement.execute_VND(cur_val, matrix) finds best solution ,also saved in matrx obj + new value of VND solution
            # if VND-value < cur_val:
                # matrix=VNd-matrix
                # k=1
            # else: k +=1

    # matrix bicluster set to subgraph and return it
    optimized_subgraph= nx.subgraph
    return optimized_subgraph