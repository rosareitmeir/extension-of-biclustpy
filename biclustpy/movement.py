import itertools

import helpers
import numpy as np
# neighborhood movements: Mov-Vertex, Join-Bicluster, Break Bicluster
# initialize matrix and update it

class Edit_matrix:
    def __init__(self,weights ,subgraph):
        self.weights=weights
        self.num_rows=weights.shape[0]
        # all nodes in the subgraph
        self.V1 = [node for node in subgraph.nodes if helpers.is_row(node, self.num_rows)]
        self.V2 = [node for node in subgraph.nodes if helpers.is_col(node, self.num_rows)]
        self.bicluster_set= helpers.connected_components(subgraph)
        self.number_biclusters=len(self.bicluster_set)
        self.edit_matrix, self.node_to_matrix = self.initialize_edit_matrix()
        #self.biclust_to_matrix= {x: self.bicluster_set[x].name for x in range(len(self.bicluster_set))}

    def initialize_edit_matrix(m):
        # according to page 10 of the paper

        # setting all entries to 0
        edit_matrix = np.zeros((len(m.V1) + len(m.V2), m.number_biclusters))
        node_to_matrix={}
        #bicluster_to_matrix={}
        # calculate entries row for row
        edit_matrix,node_to_matrix=m.fillrowbyrow(True,0, edit_matrix,node_to_matrix)
        edit_matrix,node_to_matrix=m.fillrowbyrow(False,len(m.V1),edit_matrix,node_to_matrix)

        return edit_matrix, node_to_matrix



    def fillrowbyrow(m,rightorder,counter, edit_matrix, node_to_matrix):

        # according to defintion of matrix entries on page 10
        if rightorder:
            nodes1= m.V1
            nodes2=m.V2
        else:
            nodes1 = m.V2
            nodes2 = m.V1

        for node1 in nodes1:

            for b in range(len(m.bicluster_set)): # cols of editing matrix
                # calculate sum of the edge weights between node1 € V1 and all nodes2 € (V2 and biclique b)
                for node2 in [elem for elem in nodes2 if elem in m.bicluster_set[b]]:
                    edit_matrix[counter][b]+= get_weights(rightorder, node1,node2,m.num_rows,m.weights)
                # sum is negative, if node1 is not an element of biclique b
                if node1 not in m.bicluster_set[b]:
                    edit_matrix[counter][b] *= -1

            node_to_matrix[node1]=counter
            counter += 1

        return edit_matrix,node_to_matrix



def get_weights(rightorder,node1, node2, num_rows,weights):
    if rightorder: # node1 € V1(=row) and node2 € V2(=col)
        colID = helpers.node_to_col(node2, num_rows)
        return weights[node1][colID];
    else:
        colID = helpers.node_to_col(node1, num_rows)
        return weights[node2][colID];



def execute_VND( curval, edit_matrix):
    # 0: move-vertex neighbourhood , 1: join-bicluster neighbourhood, 2: break-bicluster neighbourhood
    k=0; # -> starting with moving solitude nodes
    while k<3:
        changed=False
        # find best solution for the current neighbourhood
        # move vertex
        if k==0:
            # calc values and check it to current value
            curval,neighbour=calc_move_vertex(curval, edit_matrix)
            if neighbour != None: #new better solution found
                changed=True
                # update edit matrix and solution= bicluster_set : remove moved_vertex and add it to the other bicluster
                update_move_vertex(neighbour,edit_matrix)


        # join bicluster
        elif k==1:
            # calc values and check it to current value
            curval, neighbour=calc_join_bicluster(curval, edit_matrix)
            if neighbour!=None:
                changed=True
                # update edit matrix and bicluster_Set : remove both biclusters and add new joined one
                update_join_bicluster(neighbour,edit_matrix)


        # break bicluster
        else: #k==2
            calc_break_bicluster(curval,edit_matrix)


        # for all three neighbourhoods:
        if changed: k=1
        else : k+=1

    return edit_matrix, curval


def calc_move_vertex(curval, m): # curval is value of the current solution ,m is an object of edit-matrix class
    bestval=curval
    bestneighbour=None

    for i in range(m.number_biclusters): # loop through every bicluster
        for node in m.bicluster_set[i].nodes: # every node in this bicluster
            matrix_index= m.node_to_matrix[node]
            for j in range(m.number_biclusters): # moving to other bicluster
                if i==j: continue
                value= curval+ m.edit_matrix[matrix_index][j]+ m.edit_matrix[matrix_index][i]
                if value < bestval:
                    bestval=value
                    bestneighbour=[node,i,j] # list of movement: moved node from bicluster i to bicluster j

    return curval,bestneighbour

def update_move_vertex(neighbour, m): # neighbour with moved_node and the two biclusters, m is an object of edit-matrix class
    moved_node=neighbour[0]
    index_moved_node=m.node_to_matrix[moved_node]
    before_cluster_index=neighbour[1]
    before_cluster= m.bicluster_set[before_cluster_index]
    after_cluster_index = neighbour[2]
    after_cluster = m.bicluster_set[after_cluster_index]

    # update edit matrix
    m.edit_matrix[index_moved_node][before_cluster_index] *= -1
    m.edit_matrix[index_moved_node][after_cluster_index] *= -1

    if moved_node<= m.num_rows: # moved node element of V1/rows
        V2=m.V2
        rightorder=True

    else:
        V2=m.V1
        rightorder=False

    # entries for all nodes € V2
    for node2 in V2:
        node2_index = m.node_to_matrix[node2]
        # edge weight between moved node and node2 € V2
        edgeweight = get_weights(rightorder, moved_node, node2, m.num_rows, m.weights)
        if node2 in before_cluster.nodes:
            m.edit_matrix[node2_index][before_cluster_index] -= edgeweight
            m.edit_matrix[node2_index][after_cluster_index] -= edgeweight

        elif node2 in after_cluster.nodes:
            m.edit_matrix[node2_index][before_cluster_index] += edgeweight
            m.edit_matrix[node2_index][after_cluster_index] += edgeweight
        else:
            m.edit_matrix[node2_index][before_cluster_index] += edgeweight
            m.edit_matrix[node2_index][after_cluster_index] -= edgeweight

    # update bicluster set
    before_cluster.remove_node(moved_node)
    after_cluster.add_node(moved_node)
    after_cluster.add_edges_from([(moved_node, l)  for l in [elem for elem in after_cluster.nodes if elem in V2]])
    #check=helpers.is_bi_clique(after_cluster,m.num_rows)

    return

def calc_join_bicluster(curval, m):
    bestneighbour=None
    bestval=curval
    for biclustpair in itertools.combinations(m.bicluster_set, r=2): # calculate value for every possible bicluster join
        biclust1 = biclustpair[0]
        index_biclust1 = m.bicluster_set.index(biclust1)
        biclust2 = biclustpair[1]
        index_biclust2 = m.bicluster_set.index(biclust2)
        # calculating sum page 11 formula (11)
        sum=0
        for node in biclust1.nodes:
            sum += m.edit_matrix[m.node_to_matrix(node)][index_biclust2]
        val= curval+sum
        if val < bestval:
            bestval=val
            bestneighbour=[index_biclust1, index_biclust2]


    return bestval, bestneighbour


def update_join_bicluster(neighbour,m):
    # update edit matrix
    # update bicluster set


    return neighbour





def calc_break_bicluster(curval,m):
    return curval