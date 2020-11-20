import copy
import itertools
from . import helpers
import numpy as np
# static variables for all obj of Solution
weights=None
num_rows=None
V1=None
V2=None

class Solution:
    # object, which saves edit matrix and biclusters (graphs)
    def __init__(self,weight ,subgraph):
        global weights
        weights=weight
        global num_rows
        num_rows=weights.shape[0]
        # all nodes in the subgraph
        global V1
        global V2
        V1 = [node for node in subgraph.nodes if helpers.is_row(node, num_rows)]
        V2 = [node for node in subgraph.nodes if helpers.is_col(node, num_rows)]
        self.bicluster_set= helpers.connected_components(subgraph)
        self.number_biclusters=len(self.bicluster_set)
        self.edit_matrix, self.node_to_matrix = self.initialize_edit_matrix()

    def initialize_edit_matrix(self):
        # according to the instruction on page10
        # set all entries to zero
        edit_matrix = np.zeros((len(V1) + len(V2), self.number_biclusters))
        node1_idx=0
        node_to_matrix = {}
        # go through every pair (node1,node2) node1 € V1 , node2 € V2
        for node1 in V1:
            node2_idx = len(V1)
            i = 0
            # find biclique of node 1
            while node1 not in self.bicluster_set[i].nodes: i += 1
            biclique1 = i

            for node2 in V2:
                # find biclique of node2
                j = 0
                while node2 not in self.bicluster_set[j].nodes: j += 1
                biclique2 = j

                if biclique1==biclique2: # same cluster add conserved edges to entry
                    edit_matrix[node1_idx][biclique1]+= get_weight(True,node1,node2)
                    edit_matrix[node2_idx][biclique2] += get_weight(True,node1,node2)

                else: # different cluster subtract lost edges
                    edit_matrix[node1_idx][biclique2] -= get_weight(True,node1,node2)
                    edit_matrix[node2_idx][biclique1] -= get_weight(True,node1,node2)

                node_to_matrix[node2]=node2_idx
                node2_idx+=1

            node_to_matrix[node1] = node1_idx
            node1_idx += 1

        return edit_matrix,node_to_matrix



def get_weight(rightorder, node1, node2):
    if rightorder: # node1 € V1(=row) and node2 € V2(=col)
        colID = helpers.node_to_col(node2, num_rows)
        return weights[node1][colID];
    else:
        colID = helpers.node_to_col(node1, num_rows)
        return weights[node2][colID];



def execute_VND(curval, sol):
    ''' Variable Neighbourhood Descent
     according to page 12 and 13 and its pseudocode of Fig. 9 on page 14 '''
    VNDsolution= copy.deepcopy(sol)
    VND_val= curval
    # 0: move-vertex neighbourhood , 1: join-bicluster neighbourhood, 2: break-bicluster neighbourhood
    k=0; # -> starting with moving a single node
    while k<3:
        changed=False
        # move vertex
        if k==0:
            # find best neighbour in the neighbourhood move vertex
            neighbour_val,neighbour=find_best_move_vertex(VND_val, VNDsolution)
            if neighbour_val<VND_val: #new better solution found
                changed=True
                VND_val=neighbour_val
                # update edit matrix and bicluster set: remove moved_vertex and add it to the other bicluster
                update_move_vertex(neighbour,VNDsolution)
        # join bicluster
        elif k==1:
            # find best neighbour in the neighbourhood join bicluster
            neighbour_val, neighbour=find_best_join_bicluster(VND_val, VNDsolution)
            if neighbour_val< VND_val:
                changed=True
                VND_val=neighbour_val
                # update edit matrix and bicluster set : remove both biclusters and add new joined one
                update_join_bicluster(neighbour,VNDsolution)

        # break bicluster
        else: #k==2
            # find best neighbour in the neighbourhood break bicluster
            neighbour_val, neighbour=find_best_break_bicluster(VND_val, VNDsolution)
            if neighbour_val<VND_val:
                VND_val=neighbour_val
                # update edit matrix and bicluster set : remove broken bicluster and add the two new ones
                changed = True
                update_break_bicluster(neighbour,VNDsolution)

        # for all three neighbourhoods:
        if changed: k=0
        else : k+=1

    return VNDsolution,VND_val


def find_best_move_vertex(curval, sol): # curval is value of the current solution ,m is an object of solution class
    bestval= np.inf
    bestneighbour=None

    for i in range(sol.number_biclusters): # loop through every bicluster
        for node in sol.bicluster_set[i].nodes: # every node in this bicluster
            matrix_index= sol.node_to_matrix[node]
            partion=V1
            if helpers.is_col(node,num_rows):
                partion=V2
            for j in range(sol.number_biclusters): # moving to other bicluster
                if i==j or helpers.is_singleton_in_same_partion(sol.bicluster_set[j],partion):
                    # not a different bicluster or moving a vertex will lead to an illegal bicluster
                    continue
                # calculate objective value for the neighbour according to formula (10) on page 11
                value= curval + sol.edit_matrix[matrix_index][j] + sol.edit_matrix[matrix_index][i]
                if value < bestval:
                    bestval=value
                    bestneighbour=[node,i,j] # list of movement: moved node from bicluster i to bicluster j

    return bestval,bestneighbour

def update_move_vertex(neighbour, sol): # neighbour with moved_node and the two biclusters, m is an object of solution class
    moved_node=neighbour[0]
    index_moved_node=sol.node_to_matrix[moved_node]
    before_cluster_index=neighbour[1]
    before_cluster= sol.bicluster_set[before_cluster_index]
    after_cluster_index = neighbour[2]
    after_cluster = sol.bicluster_set[after_cluster_index]

    # update edit matrix according to page 11
    # entries of moved node
    sol.edit_matrix[index_moved_node][before_cluster_index] *= -1
    sol.edit_matrix[index_moved_node][after_cluster_index] *= -1

    if moved_node< num_rows: # moved node  is an element of V1/row
        partition2 =list(V2)
        rightorder=True

    else:
        partition2= list(V1)
        rightorder=False

    #entries for all nodes, which are in the other partition
    for node2 in partition2:
        node2_index = sol.node_to_matrix[node2]
        # edge weight between moved node and node2
        edgeweight = get_weight(rightorder, moved_node, node2)
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
    # in case moved_node was before a singleton
    if len(before_cluster.nodes)==0:
        sol.edit_matrix=np.delete(sol.edit_matrix, before_cluster_index, 1)
        sol.bicluster_set.remove(before_cluster)
        sol.number_biclusters -=1
    after_cluster.add_node(moved_node)
    after_cluster.add_edges_from([(moved_node, l)  for l in [elem for elem in after_cluster.nodes if elem in partition2]])

def find_best_join_bicluster(curval, sol):
    bestneighbour=None
    bestval= np.inf
    for biclustpair in itertools.combinations(sol.bicluster_set, r=2): # calculate value for every possible bicluster join
        biclust1 = biclustpair[0]
        index_biclust1 = sol.bicluster_set.index(biclust1)
        biclust2 = biclustpair[1]
        index_biclust2 = sol.bicluster_set.index(biclust2)
        if helpers.are_singeltons_in_same_partion(biclust1, biclust2): continue # no valid join
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

    joined_biclust= np.zeros((len(V1) + len(V2), 1))
    # calculation resp. to formula on page 12
    for node in (V1 + V2):
        matrix_index= sol.node_to_matrix[node]
        if node in biclust1:
            joined_biclust[matrix_index]= sol.edit_matrix[matrix_index][biclust1_index] - sol.edit_matrix[matrix_index][biclust2_index]
        elif node in biclust2:
            joined_biclust[matrix_index]= sol.edit_matrix[matrix_index][biclust2_index] - sol.edit_matrix[matrix_index][biclust1_index]
        else:
            joined_biclust[matrix_index]= sol.edit_matrix[matrix_index][biclust2_index] + sol.edit_matrix[matrix_index][biclust1_index]

    # removing biclusters and add new one
    sol.edit_matrix = np.delete(sol.edit_matrix, max(biclust2_index,biclust1_index), 1)
    sol.edit_matrix = np.delete(sol.edit_matrix, min(biclust2_index,biclust1_index), 1)
    sol.edit_matrix=np.append(sol.edit_matrix, joined_biclust, axis=1)

    # update bicluster set
    joined_biclust= helpers.build_bicluster(biclust1.nodes, biclust2.nodes)
    sol.bicluster_set.append(joined_biclust)
    sol.bicluster_set.remove(biclust1)
    sol.bicluster_set.remove(biclust2)
    sol.number_biclusters -=1

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
    # no break possible , all nodes in one bicluster or one of them is a non valid biclique (at least 2 nodes and all nodes are in the same partion):
    if not biclust1 or not biclust2 or helpers.is_no_valid_biclique(biclust1,biclust2):
        return None, None, None

    # calculating value for break , formula (14) on page 12
    #first sum
    sum1 = 0
    for node1 in [x for x in biclust1 if x in V1]:
        for node2 in [x for x in biclust2 if x in V2]:
            sum1 += get_weight(True, node1, node2)
    # second sum
    sum2 = 0
    for node1 in [x for x in biclust2 if x in V1]:
        for node2 in [x for x in biclust1 if x in V2]:
            sum2 += get_weight(True, node1, node2)
    # together:
    value = curval + sum1 + sum2
    return value, biclust1,biclust2



def update_break_bicluster(neighbour, sol):
    # according to page 12
    # update edit matrix
    biclust1=neighbour[0]
    biclust2=neighbour[1]
    broken_clust_idx= neighbour[2]
    # calculations for new bicluster1 and bicluster2 from broken bicluster
    # column for bicluster1:
    col_B1=np.zeros((len(V1) + len(V2 ), 1))
    col_B1=build_cluster_column(col_B1, True, biclust1, biclust2, broken_clust_idx, sol)
    col_B1=build_cluster_column(col_B1, False, biclust1, biclust2, broken_clust_idx, sol)

    # column for bicluster2
    col_B2 = np.zeros((len(V1) + len(V2), 1))
    col_B2 = build_cluster_column(col_B2, True, biclust2, biclust1, broken_clust_idx, sol)
    col_B2 = build_cluster_column(col_B2, False, biclust2, biclust1, broken_clust_idx, sol)

    sol.edit_matrix= np.append(sol.edit_matrix, col_B1, axis=1)
    sol.edit_matrix=np.append(sol.edit_matrix, col_B2, axis=1)
    sol.edit_matrix=np.delete(sol.edit_matrix, broken_clust_idx, 1)

    #update bicluster set
    bicluster1= helpers.build_bicluster( biclust1)
    bicluster2= helpers.build_bicluster( biclust2)
    sol.bicluster_set.append(bicluster1)
    sol.bicluster_set.append(bicluster2)
    sol.bicluster_set.remove(sol.bicluster_set[broken_clust_idx])
    sol.number_biclusters= len(sol.bicluster_set)



def build_cluster_column(column, rightorder, biclust1, biclust2, broken_clust_idx, sol):
    if rightorder:
        partion1=V1; partion2=V2

    else:
        partion1=V2; partion2=V1

    # calculations according to page 12
    for node1 in partion1:
        node1_index=sol.node_to_matrix[node1]
        #sum over all nodes € V2 and € biclust2(=B'')
        sum=0
        for node2 in [x for x in partion2 if x in biclust2]:
            sum += get_weight(rightorder, node1, node2)

        entry_n1_broken_biclust= sol.edit_matrix[node1_index][broken_clust_idx] # M(v,B)

        if node1 in biclust1:
            column[node1_index]= entry_n1_broken_biclust-sum
        elif node1 in biclust2:
            column[node1_index]= -1*entry_n1_broken_biclust+sum
        else:
            column[node1_index]= entry_n1_broken_biclust+sum

    return column

def shake_solution(nmin, nmax, inputsol, before_val, k=None):
    # perturbation phase for GVNS with fixed k (page 13 defined) and for ILS with random k (page 14 defined)
    shaked_sol= copy.deepcopy(inputsol)
    shaked_val=before_val
    ILS=False
    if k==None:
        k= np.random.randint(3)
        ILS=True
    # random number of perturbation movements , default between 2 and 10
    n = np.random.randint(low=nmin,high=nmax)
    i=0
    while i <n:
        # move vertex
        if k==0 and shaked_sol.number_biclusters>1:
            # find all possible and valid movements
            pos_movevertex = get_possible_movevertex(shaked_sol)
            random_movement= pos_movevertex[np.random.randint(len(pos_movevertex))]
            random_node=random_movement[0]
            before_clust=random_movement[1]
            # choose random bicluster, in which node will be moved.
            after_clust=int(np.random.choice(random_movement[2],1))
            node_idx=shaked_sol.node_to_matrix[random_node]
            #calc value for new pertubated solution
            shaked_val = shaked_val + shaked_sol.edit_matrix[node_idx][after_clust] + shaked_sol.edit_matrix[node_idx][before_clust]
            # update edit matrix and bicluster set of the solution
            neighbour= [random_node,before_clust,after_clust]
            update_move_vertex(neighbour,shaked_sol)

        # join bicluster
        elif k==1 and shaked_sol.number_biclusters>1:
            # choosing randomly two biclusters
            neighbour= np.random.choice(shaked_sol.number_biclusters,2,replace=False)
            biclust1 = shaked_sol.bicluster_set[neighbour[0]]
            biclust2 = shaked_sol.bicluster_set[neighbour[1]]
            while  helpers.are_singeltons_in_same_partion(biclust1,biclust2): # two singletons in same partition can't be joined
                neighbour = np.random.choice(shaked_sol.number_biclusters, 2, replace=False)
                biclust1 = shaked_sol.bicluster_set[neighbour[0]]
                biclust2 = shaked_sol.bicluster_set[neighbour[1]]

            shaked_val= calc_join_bicluster(biclust1,neighbour[1],shaked_sol,shaked_val)
            update_join_bicluster(neighbour, shaked_sol)

        # break bicluster
        elif ILS or k==2:
                pos_movements=get_possible_breakcluster(shaked_sol,shaked_val)
                if not pos_movements: # rare case: only singletons in current solution or non valid breaks possible resulted from bind function
                    if ILS and shaked_sol.number_biclusters>1: # try it again with move-vertex or join-bicluster
                        k = np.random.randint(3)
                        continue
                    else: break # GVNS case, no valid neighbour in break-bicluster-neighbourhood -> shaking is finished
                break_movement=pos_movements[np.random.randint(len(pos_movements))]
                shaked_val= break_movement[0]
                neighbour=break_movement[1]
                update_break_bicluster(neighbour,shaked_sol)
        else: break # case GVNS and no possible neighbour in the fixed neighbourhood move-vertex/join-cluster -> shaking is finished
        if ILS:
            k=np.random.randint(3)
        i+=1
    return shaked_sol,shaked_val

def get_possible_movevertex(sol):
    node_afterclusters=[]
    for node in (V1+V2):
        partion=V1
        if helpers.is_col(node,num_rows): partion=V2
        # find biclique of current node
        beforeclust = 0
        while node not in sol.bicluster_set[beforeclust].nodes:
            beforeclust += 1
        # collect every possible biclique for node (not the same and no singelton of same partion)
        possible_aftercluster = []
        for i in range(sol.number_biclusters):
            biclust= sol.bicluster_set[i]
            if i == beforeclust or helpers.is_singleton_in_same_partion(biclust,partion):
                continue
            possible_aftercluster.append(i)

        if len( possible_aftercluster)>0:
            node_afterclusters.append([node,beforeclust,possible_aftercluster])

    return node_afterclusters

def get_possible_breakcluster(sol,value):
    possible_break_clusters=[]
    for i in range(sol.number_biclusters):
        checked_val, biclust1, biclust2 = calc_break_bicluster(i, sol, value)
        if checked_val!=None:
            possible_break_clusters.append([checked_val,[biclust1,biclust2,i]])

    return possible_break_clusters












