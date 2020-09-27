import copy
import itertools

import helpers
import numpy as np
# neighborhood movements: Mov-Vertex, Join-Bicluster, Break Bicluster
# initialize matrix and update it

class Solution:
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


    def initialize_edit_matrix(self):
        node_to_matrix={}
        # according to the instruction on page10
        # set all entries to zero
        edit_matrix = np.zeros((len(self.V1) + len(self.V2), self.number_biclusters))
        node1_idx=0
        i = 0
        # go through every pair (node1,node2) node1 € V1 , node2 € V2
        for node1 in self.V1:
            node2_idx = len(self.V1)
            j = 0
            # find biclique of node 1
            while node1 not in self.bicluster_set[i].nodes: i += 1
            biclique1 = i

            for node2 in self.V2:
                # find biclique of node2
                while node2 not in self.bicluster_set[j].nodes: j += 1
                biclique2 = j

                if biclique1==biclique2: # same cluster add conserved edges to entry
                    edit_matrix[node1_idx][biclique1]+= get_weight(True,node1,node2,self.num_rows,self.weights)
                    edit_matrix[node2_idx][biclique2] += get_weight(True,node1,node2,self.num_rows,self.weights)

                else: # different cluster  substract lost edges
                    edit_matrix[node1_idx][biclique2] -= get_weight(True,node1,node2,self.num_rows,self.weights)
                    edit_matrix[node2_idx][biclique1] -= get_weight(True,node1,node2,self.num_rows,self.weights)

                node_to_matrix[node2]=node2_idx
                node2_idx+=1

            node_to_matrix[node1] = node1_idx
            node1_idx += 1

        return edit_matrix,node_to_matrix



def get_weight(rightorder, node1, node2, num_rows, weights):
    if rightorder: # node1 € V1(=row) and node2 € V2(=col)
        colID = helpers.node_to_col(node2, num_rows)
        return weights[node1][colID];
    else:
        colID = helpers.node_to_col(node1, num_rows)
        return weights[node2][colID];



def execute_VND(curval, sol):
    VNDsolution= copy.deepcopy(sol)
    VND_val= curval
    # 0: move-vertex neighbourhood , 1: join-bicluster neighbourhood, 2: break-bicluster neighbourhood
    k=0; # -> starting with moving solitude node
    while k<3:
        changed=False
        # find best solution for the current neighbourhood
        # move vertex
        if k==0:
            # find best neighbour in the neighbourhood move vertex
            neighbour_val,neighbour=find_best_move_vertex(VND_val, VNDsolution)
            if neighbour_val<VND_val: #new better solution found
                changed=True
                VND_val=neighbour_val
                # update edit matrix and solution= bicluster_set : remove moved_vertex and add it to the other bicluster
                update_move_vertex(neighbour,VNDsolution)
        # join bicluster
        elif k==1:
            # find best neighbour in the neighbourhood join bicluster
            neighbour_val, neighbour=find_best_join_bicluster(VND_val, VNDsolution)
            if neighbour_val< VND_val:
                changed=True
                VND_val=neighbour_val
                # update edit matrix and bicluster_Set : remove both biclusters and add new joined one
                update_join_bicluster(neighbour,VNDsolution)

        # break bicluster
        else: #k==2
            # find best neighbour in the neighbourhood break bicluster
            neighbour_val, neighbour=find_best_break_bicluster(VND_val, VNDsolution)
            if neighbour_val<VND_val:
                VND_val=neighbour_val
                # update edit matrix and bicluster_Set : remove broken biclusters and add the two new ones
                changed = True
                update_break_bicluster(neighbour,VNDsolution)

        # for all three neighbourhoods:
        if changed: k=1
        else : k+=1

    return VNDsolution,VND_val


def find_best_move_vertex(curval, sol): # curval is value of the current solution ,m is an object of edit-matrix class
    bestval= np.inf
    bestneighbour=None

    for i in range(sol.number_biclusters): # loop through every bicluster
        for node in sol.bicluster_set[i].nodes: # every node in this bicluster
            matrix_index= sol.node_to_matrix[node]
            for j in range(sol.number_biclusters): # moving to other bicluster
                if i==j: continue
                value= curval + sol.edit_matrix[matrix_index][j] + sol.edit_matrix[matrix_index][i]
                if value < bestval:
                    bestval=value
                    bestneighbour=[node,i,j] # list of movement: moved node from bicluster i to bicluster j

    return bestval,bestneighbour

def update_move_vertex(neighbour, sol): # neighbour with moved_node and the two biclusters, m is an object of edit-matrix class
    moved_node=neighbour[0]
    index_moved_node=sol.node_to_matrix[moved_node]
    before_cluster_index=neighbour[1]
    before_cluster= sol.bicluster_set[before_cluster_index]
    after_cluster_index = neighbour[2]
    after_cluster = sol.bicluster_set[after_cluster_index]

    # update edit matrix
    sol.edit_matrix[index_moved_node][before_cluster_index] *= -1
    sol.edit_matrix[index_moved_node][after_cluster_index] *= -1

    if moved_node< sol.num_rows: # moved node element of V1/rows
        V2=sol.V2
        rightorder=True

    else:
        V2=sol.V1
        rightorder=False

    # entries for all nodes € V2
    for node2 in V2:
        node2_index = sol.node_to_matrix[node2]
        # edge weight between moved node and node2 € V2
        edgeweight = get_weight(rightorder, moved_node, node2, sol.num_rows, sol.weights)
        if node2 in before_cluster.nodes:
            sol.edit_matrix[node2_index][before_cluster_index] -= edgeweight
            sol.edit_matrix[node2_index][after_cluster_index] -= edgeweight

        elif node2 in after_cluster.nodes:
            sol.edit_matrix[node2_index][before_cluster_index] += edgeweight
            sol.edit_matrix[node2_index][after_cluster_index] += edgeweight
        else:
            sol.edit_matrix[node2_index][before_cluster_index] += edgeweight
            sol.edit_matrix[node2_index][after_cluster_index] -= edgeweight

    # update bicluster set
    before_cluster.remove_node(moved_node)
    after_cluster.add_node(moved_node)
    after_cluster.add_edges_from([(moved_node, l)  for l in [elem for elem in after_cluster.nodes if elem in V2]])
    #check=helpers.is_bi_clique(after_cluster,m.num_rows)

    return

def find_best_join_bicluster(curval, sol):
    bestneighbour=None
    bestval= np.inf
    for biclustpair in itertools.combinations(sol.bicluster_set, r=2): # calculate value for every possible bicluster join
        biclust1 = biclustpair[0]
        index_biclust1 = sol.bicluster_set.index(biclust1)
        biclust2 = biclustpair[1]
        index_biclust2 = sol.bicluster_set.index(biclust2)
        # calculate value of this solution
        val= calc_join_bicluster(biclust1,index_biclust2,sol,curval)
        if val < bestval:
            bestval=val
            bestneighbour=[index_biclust1, index_biclust2]


    return bestval, bestneighbour

def calc_join_bicluster(biclust1,index_biclust2,sol,curval):
    # calculating sum page 11 formula (11)
    sum = 0
    for node in biclust1.nodes:
        sum += sol.edit_matrix[sol.node_to_matrix[node]][index_biclust2]
    val = curval + sum
    return val


def update_join_bicluster(neighbour, sol):
    # update edit matrix
    biclust1_index=neighbour[0]
    biclust1=sol.bicluster_set[biclust1_index]
    biclust2_index = neighbour[1]
    biclust2=sol.bicluster_set[biclust2_index]

    joined_biclust= np.zeros((len(sol.V1) + len(sol.V2), 1))
    # calcualtion resp. to formula on page 12
    for node in (sol.V1 + sol.V2):
        matrix_index= sol.node_to_matrix[node]
        if node in biclust1:
            joined_biclust[matrix_index]= sol.edit_matrix[matrix_index][biclust1_index] - sol.edit_matrix[matrix_index][biclust2_index]
        elif node in biclust2:
            joined_biclust[matrix_index]= sol.edit_matrix[matrix_index][biclust2_index] - sol.edit_matrix[matrix_index][biclust1_index]
        else:
            joined_biclust[matrix_index]= sol.edit_matrix[matrix_index][biclust2_index] + sol.edit_matrix[matrix_index][biclust1_index]

    sol.edit_matrix=np.append(sol.edit_matrix, joined_biclust, axis=1)
    sol.edit_matrix = np.delete(sol.edit_matrix, biclust1_index, 1)
    sol.edit_matrix = np.delete(sol.edit_matrix, biclust2_index, 1)

    # update bicluster set
    joined_biclust= helpers.build_bicluster(sol, biclust1.nodes, biclust2.nodes)
    sol.bicluster_set.append(joined_biclust)
    sol.bicluster_set.remove(biclust1)
    sol.bicluster_set.remove(biclust2)
    sol.number_biclusters= len(sol.bicluster_set)

def find_best_break_bicluster(curval, sol):
    bestvalue=np.inf
    bestneighbour=None
    # according to page 12
    for i in range(len(sol.bicluster_set)):
        # calculate value for this solution
        value,biclust1,biclust2=calc_break_bicluster(i,sol,curval)
        if value==None:
            continue
        if value<bestvalue:
            bestvalue=value
            bestneighbour=[biclust1,biclust2,i]

    return bestvalue,bestneighbour


def calc_break_bicluster(biclust_idx,sol,curval):
    bicluster = sol.bicluster_set[biclust_idx]
    biclust1 = []
    biclust2 = []
    #  sorting nodes in biclust1 and biclust2 considering their bind function value
    for node in bicluster.nodes:
        node_index = sol.node_to_matrix[node]
        if sol.edit_matrix[node_index][biclust_idx] < 0:
            biclust1.append(node)
        else:
            biclust2.append(node)
    # no break possible , all nodes in one bicluster:
    if not biclust1 or not biclust2:
        return None, None, None

    # calculating value for break , formula (14) on page 12
    #first sum
    sum1 = 0
    for node1 in [x for x in biclust1 if x in sol.V1]:
        for node2 in [x for x in biclust2 if x in sol.V2]:
            sum1 += get_weight(True, node1, node2, sol.num_rows, sol.weights)
    # second sum
    sum2 = 0
    for node1 in [x for x in biclust2 if x in sol.V1]:
        for node2 in [x for x in biclust1 if x in sol.V2]:
            sum2 += get_weight(True, node1, node2, sol.num_rows, sol.weights)
    # together:
    value = curval + sum1 + sum2
    return value, biclust1,biclust2



def update_break_bicluster(neighbour, sol):
    # update edit matrix
    biclust1=neighbour[0]
    biclust2=neighbour[1]
    broken_clust_idx= neighbour[2]
    # calculations for new bicluster1 and bicluster2 from broken bicluster
    # column for bicluster1:
    col_B1=np.zeros((len(sol.V1) + len(sol.V2), 1))
    col_B1=build_cluster_column(col_B1, True, biclust1, biclust2, broken_clust_idx, sol)
    col_B1=build_cluster_column(col_B1, False, biclust1, biclust2, broken_clust_idx, sol)

    # column for bicluster2
    col_B2 = np.zeros((len(sol.V1) + len(sol.V2), 1))
    col_B2 = build_cluster_column(col_B2, True, biclust2, biclust1, broken_clust_idx, sol)
    col_B2 = build_cluster_column(col_B2, False, biclust2, biclust1, broken_clust_idx, sol)

    sol.edit_matrix= np.append(sol.edit_matrix, col_B1, axis=1)
    sol.edit_matrix=np.append(sol.edit_matrix, col_B2, axis=1)
    sol.edit_matrix=np.delete(sol.edit_matrix, broken_clust_idx, 1)

    #update bicluster set
    bicluster1= helpers.build_bicluster(sol, biclust1)
    bicluster2= helpers.build_bicluster(sol, biclust2)
    sol.bicluster_set.append(bicluster1)
    sol.bicluster_set.append(bicluster2)
    sol.bicluster_set.remove(sol.bicluster_set[broken_clust_idx])
    sol.number_biclusters= len(sol.bicluster_set)



def build_cluster_column(column, rightorder, biclust1, biclust2, broken_clust_idx, sol):
    if rightorder: V1=sol.V1; V2=sol.V2
    else: V1=sol.V2; V2=sol.V1
    # calculations according to page 12
    for node1 in V1:
        node1_index=sol.node_to_matrix[node1]
        #sum over all nodes € V2 and € biclust2(=B'')
        sum=0
        for node2 in [x for x in V2 if x in biclust2]:
            sum += get_weight(rightorder, node1, node2, sol.num_rows, sol.weights)

        entry_n1_broken_biclust= sol.edit_matrix[node1_index][broken_clust_idx] # M(v,B)

        if node1 in biclust1:
            column[node1_index]= entry_n1_broken_biclust-sum
        elif node1 in biclust2:
            column[node1_index]= -entry_n1_broken_biclust+sum
        else:
            column[node1_index]= entry_n1_broken_biclust+sum

    return column

def shake_solution(nmin, nmax, inputsol, before_val, k=None):
    shaked_sol= copy.deepcopy(inputsol)
    shaked_val=before_val
    ILS=False
    if k==None:
        k= np.random.randint(3)
        ILS=True
    n = np.random.randint(low=nmin,high=nmax)
    for i in range(n):
        # move vertex
        if k==0 and shaked_sol.number_biclusters>1:
            random_node= np.random.choice(shaked_sol.V1 + shaked_sol.V2)
            # find its bicluster 
            for before_clust in range(shaked_sol.number_biclusters):
                if random_node in shaked_sol.bicluster_set[before_clust].nodes:
                    after_clust=np.random.randint(shaked_sol.number_biclusters)
                    while after_clust==before_clust:
                        after_clust=np.random.randint(shaked_sol.number_biclusters)
                    node_idx=shaked_sol.node_to_matrix[random_node]
                    #calc value for new pertubated solution
                    shaked_val = shaked_val + shaked_sol.edit_matrix[node_idx][after_clust] + shaked_sol.edit_matrix[node_idx][before_clust]
                    # update edit matrix and bicluster set of the solution
                    neighbour= [random_node,before_clust,after_clust]
                    update_move_vertex(neighbour,shaked_sol)
                    break
        # join bicluster
        elif k==1 and shaked_sol.number_biclusters>1:
            # choosing randomly two biclusters
            neighbour= np.random.choice(shaked_sol.number_biclusters,2,replace=False)
            biclust1=shaked_sol.bicluster_set[neighbour[0]]
            shaked_val= calc_join_bicluster(biclust1,neighbour[1],shaked_sol,shaked_val)
            update_join_bicluster(neighbour, shaked_sol)
        # break bicluster
        elif ILS or k==2:
            broken_idx=np.random.randint(shaked_sol.number_biclusters)
            checked_val,biclust1,biclust2= calc_break_bicluster(broken_idx,shaked_sol,shaked_val)
            if checked_val!=None:
                shaked_val=checked_val
                neighbour=[biclust1,biclust2,broken_idx]
                update_break_bicluster(neighbour,shaked_sol)
        else: break # case GVNS and no possible neighbour in the fixed neighbourhood move-vertex/join-cluster -> shaking is finished
        if ILS:
            k=np.random.randint(3)

    return shaked_sol,shaked_val



