import copy
import itertools
import time
import random

#from .
import helpers
import numpy as np
import random
import networkx as nx
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


def run_VND(weights, subgraph, obj_val):
    initialized_solution = Solution(weights, subgraph)
    solution, value = execute_VND(obj_val, initialized_solution)
    optimized_subgraph = helpers.graph_from_components(solution.bicluster_set)
    return optimized_subgraph, value, False, None

def execute_VND(curval, sol):
    ''' Variable Neighbourhood Descent
     according to page 12 and 13 and its pseudocode of Fig. 9 on page 14 '''
    VNDsolution= sol
    VND_val= curval
    # 0: move-vertex neighbourhood , 1: join-bicluster neighbourhood, 2: break-bicluster neighbourhood
    k=0; # -> starting with moving a single node
    while k<3:
        changed=False
        # move vertex
        if k==0:
            # find best neighbour in the neighbourhood move vertex
            start=time.time()
            neighbour_val,neighbour=find_best_move_vertex(VND_val, VNDsolution)

            if  VND_val- neighbour_val > 0.001: #new better solution found
                changed=True
                VND_val=neighbour_val
                # update edit matrix and bicluster set: remove moved_vertex and add it to the other bicluster
                update_move_vertex(neighbour,VNDsolution)
            #print("finished with move vertex: " +str(time.time()-start))
        # join bicluster
        elif k==1:
            # find best neighbour in the neighbourhood join bicluster
            start=time.time()
            neighbour_val, neighbour=find_best_join_bicluster(VND_val, VNDsolution)

            if VND_val- neighbour_val > 0.001:
                changed=True
                VND_val=neighbour_val
                # update edit matrix and bicluster set : remove both biclusters and add new joined one
                update_join_bicluster(neighbour,VNDsolution)
            #print("finished with join: " +str(time.time()-start))

        # break bicluster
        else: #k==2
            # find best neighbour in the neighbourhood break bicluster
            start=time.time()
            neighbour_val, neighbour=find_best_break_bicluster(VND_val, VNDsolution)
            if VND_val- neighbour_val > 0.001:
                VND_val=neighbour_val
                # update edit matrix and bicluster set : remove broken bicluster and add the two new ones
                changed = True
                update_break_bicluster(neighbour,VNDsolution)
            #int("finished with break: " +str(time.time()-start))
        # for all three neighbourhoods:
        if changed: k=0
        else : k+=1

    return VNDsolution,VND_val


def find_best_move_vertex(curval, sol): # curval is value of the current solution ,m is an object of solution class
    bestval= np.inf
    bestneighbour=None
    pos_after_clust_V1, pos_after_clust_V2= helpers.find_possible_biclusters(sol.bicluster_set, sol.number_biclusters, num_rows)

    for i in range(sol.number_biclusters): # loop through every bicluster
        for node in sol.bicluster_set[i].nodes: # every node in this bicluster
            matrix_index= sol.node_to_matrix[node]
            pos_after_clusters = pos_after_clust_V1
            if helpers.is_col(node,num_rows):
                pos_after_clusters=pos_after_clust_V2
            for j in pos_after_clusters: # moving to other bicluster
                if j!=i:
                # calculate objective value for the neighbour according to formula (10) on page 11
                    value= curval + sol.edit_matrix[matrix_index][j] + sol.edit_matrix[matrix_index][i]
                    if  bestval -value > 0.001:
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

    if moved_node< num_rows: # moved node  is an element of V1/row
        partition2 =list(V2)
        rightorder=True

    else:
        partition2= list(V1)
        rightorder=False


    # 1. update edit matrix according to page 11

    # entries of moved node
    sol.edit_matrix[index_moved_node][before_cluster_index] *= -1
    sol.edit_matrix[index_moved_node][after_cluster_index] *= -1

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

    # 2. update bicluster set
    # move node
    before_cluster.remove_node(moved_node)
    after_cluster.add_node(moved_node)
    # add edges for new node in after cluster
    if moved_node in V1:
        after_cluster.add_edges_from([(moved_node, l) for l in [elem for elem in after_cluster.nodes if elem in partition2]])
    else:
        after_cluster.add_edges_from([(l, moved_node) for l in [elem for elem in after_cluster.nodes if elem in partition2]])

    # special case 1: moved node was only node from a different partition -> before cluster consists of singletons
    if all(n in partition2 for n in before_cluster.nodes) and len(before_cluster.nodes) > 1:
        # before cluster must be transformed to singletons with own bicluster column
        for node in list(before_cluster.nodes):
            column = build_singelton_column(node, sol,  rightorder)
            sol.edit_matrix = np.append(sol.edit_matrix, column, axis=1)
            sol.bicluster_set.append(helpers.build_bicluster([node]))

        sol.edit_matrix = np.delete(sol.edit_matrix, before_cluster_index, 1)
        sol.bicluster_set.remove(before_cluster)
        sol.number_biclusters += len(before_cluster.nodes) - 1


    # special case 2:  before cluster is a singleton
    if len(before_cluster.nodes)==0:
        # remove cluster from set and matrix
        sol.edit_matrix=np.delete(sol.edit_matrix, before_cluster_index, 1)
        sol.bicluster_set.remove(before_cluster)
        sol.number_biclusters -=1



def build_singelton_column(single, sol,  rightorder):

    partition = V2
    if rightorder:
        partition=V1

    column=np.zeros((len(V1) + len(V2 ), 1))

    for node in partition:
        node_idx=sol.node_to_matrix[node]
        column[node_idx] -= get_weight(rightorder, node, single)

    return column


def find_best_join_bicluster(curval, sol):
    bestneighbour=None
    bestval= np.inf
    
    # find all possible bicluster join
    all_joins=get_all_possible_join_bicluster(sol)

    # calculate value for every possible bicluster join
    for (index_biclust1, index_biclust2) in all_joins:
        biclust1=sol.bicluster_set[index_biclust1]
        val= calc_join_bicluster(biclust1,index_biclust2,sol,curval)
        if  bestval -val > 0.001:
            bestval=val
            bestneighbour=[index_biclust1, index_biclust2]

    return bestval, bestneighbour


def get_all_possible_join_bicluster(sol):
    pos_after_clust_V1, pos_after_clust_V2= helpers.find_possible_biclusters(sol.bicluster_set, sol.number_biclusters, num_rows)
    V1_singletons = set(range(sol.number_biclusters)).difference(pos_after_clust_V1)
    V2_singletons = set(range(sol.number_biclusters)).difference(pos_after_clust_V2)
    other_bic = set(range(sol.number_biclusters)).difference(V1_singletons.union(V2_singletons))
    all_joins = list(itertools.product(V1_singletons, pos_after_clust_V1))
    all_joins += list(itertools.product(V2_singletons, pos_after_clust_V2))
    all_joins += list(itertools.combinations(other_bic, r=2))
    return all_joins

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

    bic2_left_nodes = [k for k in biclust2.nodes if helpers.is_row(k, num_rows)]
    bic1_left_nodes = [k for k in biclust1.nodes if helpers.is_row(k, num_rows)]
    bic2_right_nodes = [k for k in biclust2.nodes if helpers.is_col(k, num_rows)]
    bic1_right_nodes = [k for k in biclust1.nodes if helpers.is_col(k, num_rows)]


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
    joined_biclust=nx.Graph()
    joined_biclust.add_nodes_from(biclust1.nodes)
    joined_biclust.add_nodes_from(biclust2.nodes)
    nodes_v1= [e for e in joined_biclust.nodes if helpers.is_row(e,num_rows) ]
    nodes_v2= [e for e in joined_biclust.nodes if helpers.is_col(e,num_rows) ]
    joined_biclust.add_edges_from((x,y) for x in nodes_v1 for y in nodes_v2)

    sol.bicluster_set.remove(biclust1)
    sol.bicluster_set.remove(biclust2)
    sol.bicluster_set.append(joined_biclust)
    sol.number_biclusters -=1


def find_best_break_bicluster(curval, sol):
    bestvalue=np.inf
    bestneighbour=None
    # according to page 12
    for i in range(len(sol.bicluster_set)):
        # calculate value for this solution
        value,biclust1,biclust2=check_calc_break_bicluster(i, sol, curval)
        if value==None:
            continue
        if  bestvalue -value > 0.001:
            bestvalue=value
            bestneighbour=[biclust1,biclust2,i]

    return bestvalue,bestneighbour


def check_calc_break_bicluster(biclust_idx, sol, curval):
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

    value=calc_break_bicluster(biclust1, biclust2, curval)
    return value, biclust1,biclust2


def calc_break_bicluster(biclust1, biclust2, curval):
    # calculating value for break , formula (14) on page 12
    # first sum
    nodes_bic2_V2 = [x for x in biclust2 if x in V2]
    nodes_bic1_V2 = [x for x in biclust1 if x in V2]
    sum1 = 0
    for node1 in [x for x in biclust1 if x in V1]:
        for node2 in nodes_bic2_V2:
            sum1 += get_weight(True, node1, node2)
    # second sum
    sum2 = 0
    for node1 in [x for x in biclust2 if x in V1]:
        for node2 in nodes_bic1_V2:
            sum2 += get_weight(True, node1, node2)
    # together:
    value = curval + sum1 + sum2

    return value

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
    bic2_left_nodes= [k for k in biclust2 if helpers.is_row(k,num_rows)]
    bic1_left_nodes = [k for k in biclust1 if helpers.is_row(k, num_rows)]
    bic2_right_nodes = [k for k in biclust2 if helpers.is_col(k, num_rows)]
    bic1_right_nodes = [k for k in biclust1 if helpers.is_col(k, num_rows)]

    bicluster1= helpers.build_bicluster( biclust1)
    bicluster2= helpers.build_bicluster( biclust2)
    sol.bicluster_set.append(bicluster1)
    sol.bicluster_set.append(bicluster2)
    sol.bicluster_set.remove(sol.bicluster_set[broken_clust_idx])
    sol.number_biclusters= len(sol.bicluster_set)

def build_cluster_column(column, rightorder, biclust1, biclust2, broken_clust_idx, sol):
    if rightorder:
        partion1=V1
        nodes_part2_bic2 = [x for x in biclust2 if helpers.is_col(x, num_rows)]
    else:
        partion1=V2;
        nodes_part2_bic2= [x for x in biclust2 if helpers.is_row(x, num_rows)]


    # calculations according to page 12

    for node1 in partion1:
        node1_index=sol.node_to_matrix[node1]
        #sum over all nodes not € partion1 and € biclust2(=B'')
        sum=0
        for node2 in nodes_part2_bic2:
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
            # choose randomly a movement
            neighbour= get_random_mov_vertex_movement(shaked_sol)
            random_node=neighbour[0]
            before_clust=neighbour[1]
            after_clust=neighbour[2]
            node_idx=shaked_sol.node_to_matrix[random_node]
            #calc value for new pertubated solution
            shaked_val = shaked_val + shaked_sol.edit_matrix[node_idx][after_clust] + shaked_sol.edit_matrix[node_idx][before_clust]
            # update edit matrix and bicluster set of the solution
            update_move_vertex(neighbour,shaked_sol)
        # join bicluster
        elif k==1 and shaked_sol.number_biclusters>1:
            # choosing randomly two biclusters
            all_joins= get_all_possible_join_bicluster(shaked_sol)
            neighbour= list(random.choice(all_joins))
            biclust1 = shaked_sol.bicluster_set[neighbour[0]]
            shaked_val= calc_join_bicluster(biclust1,neighbour[1],shaked_sol,shaked_val)
            update_join_bicluster(neighbour, shaked_sol)

        # break bicluster
        elif ILS or k==2:
                random_movement=get_random_breakcluster(shaked_sol,shaked_val)
                if random_movement == None: # rare case: only singletons in current solution
                    if ILS and shaked_sol.number_biclusters>1: # try it again with move-vertex or join-bicluster
                        k = np.random.randint(2) # k= 0 or 1
                        continue
                    else: break # GVNS case, no valid neighbour in break-bicluster-neighbourhood -> shaking is finished
                shaked_val= random_movement[0]
                neighbour=random_movement[1]
                update_break_bicluster(neighbour,shaked_sol)

        else: break # case GVNS and no possible neighbour in the fixed neighbourhood move-vertex/join-cluster -> shaking is finished
        if ILS:
            k=np.random.randint(3)
        i+=1
    return shaked_sol,shaked_val


def get_random_breakcluster(sol,curval):
    
    not_singeltons= helpers.sort_out_singeltons(sol.bicluster_set, sol.number_biclusters)
    if len(not_singeltons)==0:
        return None
    i=random.choice(not_singeltons)
    bicluster = sol.bicluster_set[i]
    left_nodes = set([k for k in bicluster.nodes if helpers.is_row(k, num_rows)])
    right_nodes = set([k for k in bicluster.nodes if helpers.is_col(k, num_rows)])
    left_size=len(left_nodes)
    right_size=len(right_nodes)
    bc1_left = np.random.randint(left_size +1)
    bc1_right = np.random.randint(right_size + 1)

    while (bc1_left == 0 and  bc1_right != 1) or (bc1_right == 0 and bc1_left !=1) or (bc1_right==right_size and bc1_left!=left_size-1) or (bc1_left==left_size and bc1_right!=right_size-1) :
        bc1_left = np.random.randint(left_size + 1)
        bc1_right = np.random.randint(right_size + 1)

    bic1_left_nodes = set(random.sample(left_nodes, bc1_left))
    bic1_right_nodes = set(random.sample(right_nodes, bc1_right))
    biclust1= list(bic1_right_nodes) + list(bic1_left_nodes)

    bic2_left_nodes=left_nodes-bic1_left_nodes
    bic2_right_nodes=right_nodes-bic1_right_nodes
    biclust2= list(bic2_right_nodes) + list(bic2_left_nodes)

    value=calc_break_bicluster(biclust1, biclust2, curval)
    
    random_break_movement=[value, [biclust1,biclust2,i]]
    return random_break_movement
    

def get_random_mov_vertex_movement(sol):
    pos_after_clust_V1,pos_after_clust_V2= helpers.find_possible_biclusters(sol.bicluster_set,sol.number_biclusters, num_rows)

    # get all nodes, which can be moved
    all_nodes=set(V1+V2)
    if len(pos_after_clust_V1)==1: # other biclusters are singleton of same partition -> no legal movement for these nodes
        not_mov_nodes=set([ k for k  in sol.bicluster_set[pos_after_clust_V1[0]].nodes if  helpers.is_row(k, num_rows)])
        all_nodes = all_nodes - not_mov_nodes
    if len(pos_after_clust_V2) == 1:
        not_mov_nodes = set([ k for k  in sol.bicluster_set[pos_after_clust_V2[0]].nodes if helpers.is_col(k, num_rows)])
        all_nodes = all_nodes - not_mov_nodes


    random_node = random.choice(list(all_nodes))
    # find biclique of random node
    beforeclust = 0
    while random_node not in sol.bicluster_set[beforeclust].nodes:
        beforeclust += 1
    # choose random after cluster
    pos_after_clusters = set(pos_after_clust_V1)
    if helpers.is_col(random_node, num_rows): pos_after_clusters = set(pos_after_clust_V2)
    pos_after_clusters.discard(beforeclust)
    after_cluster=random.choice(list(pos_after_clusters))

    return [random_node, beforeclust, after_cluster]