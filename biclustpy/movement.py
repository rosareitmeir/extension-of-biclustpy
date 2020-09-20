import helpers
import numpy as np
# neighborhood movements: Mov-Vertex, Join-Bicluster, Break Bicluster
# initialize matrix and update it

def initialize_edit_matrix(weights, subgraph):
    # according to page 10 of the paper
    bicluster_set= helpers.connected_components(subgraph)
    number_of_biclusters= len(bicluster_set)
    num_rows = weights.shape[0]

    # all nodes in the subgraph
    V1 = [node for node in subgraph.nodes if helpers.is_row(node, num_rows)]
    V2 = [node for node in subgraph.nodes if helpers.is_col(node, num_rows)]

    # setting all entries to 0
    edit_matrix = np.zeros((len(V1) + len(V2), number_of_biclusters))

    # calculate entries row for row
    edit_matrix=fillrowbyrow(V1,V2,bicluster_set,edit_matrix, 0,weights,True,num_rows )
    edit_matrix=fillrowbyrow(V2, V1, bicluster_set, edit_matrix, len(V1),weights,False,num_rows)

    return edit_matrix



def fillrowbyrow(V1,V2,bicluster_set,edit_matrix, counter, weights, rightorder, num_rows):
    # according to defintion of matrix entries on page 10

    for node1 in V1:
        for b in range(len(bicluster_set)): # cols of editing matrix
            # calculate sum of the edge weights between node1 € V1 and all nodes2 € (V2 and Bicluster b)
            for node2 in [elem for elem in V2 if elem in bicluster_set[b]]:
                edit_matrix[counter][b]+= get_weights(rightorder, node1,node2,num_rows,weights)

            if node1 not in bicluster_set[b]:
                edit_matrix[counter][b] *= -1

        counter += 1

    return edit_matrix



def get_weights(rightorder,node1, node2, num_rows,weights):
    if rightorder:
        colID = helpers.node_to_col(node2, num_rows)
        return weights[node1][colID];
    else:
        colID = helpers.node_to_col(node1, num_rows)
        return weights[node2][colID];


