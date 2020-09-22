import networkx as nx
#import matplotlib.pyplot as plt
import movement


def run(weights, bi_transitive_subgrpah, curval):

    edit_matrix= movement.Edit_matrix(weights,bi_transitive_subgrpah)
    movement.execute_VND( curval,edit_matrix)


    optimized_subgraph= nx.subgraph
    return optimized_subgraph