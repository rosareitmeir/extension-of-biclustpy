import numpy as np
import helpers
from random import sample
import networkx as nx
def run(weights, subgrpah, num_init, metaheuristic):
    best_solution=None
    best_obj_val= np.inf
    i=0
    while (i < num_init):
        print(i)
        # create random solution
        random_sol=create_random_solution(weights,subgrpah)
        # calculate obj value of it
        obj_val=calculate_obj_val(weights, random_sol)
        # optimize it with metaheuristic
        optim_sol,optim_obj_val, optimal, time_to_best=metaheuristic.run(weights, random_sol, obj_val)
        # check if its better than current solution
        if optim_obj_val < best_obj_val:
            best_obj_val=optim_obj_val
            best_solution=optim_sol

        i+=1

    return best_solution, best_obj_val, False, None



def create_random_solution(weights, graph):
    num_rows = weights.shape[0]
    rows = set([node for node in graph.nodes if helpers.is_row(node, num_rows)])
    cols = set([node for node in graph.nodes if helpers.is_col(node, num_rows)])
    bi_transitive_subgraph = nx.Graph()

    while (len(rows)>0 and len(cols)>0):
        r=np.random.randint(len(rows)+1)
        c = np.random.randint(len(cols)+1)
        while ( r==0 and c!=1) or ( c==0 and r!=1):
            r = np.random.randint(len(rows)+1)
            c = np.random.randint(len(cols)+1)

        random_rows = set(sample(rows, r))
        random_cols = set(sample(cols, c))
        bi_transitive_subgraph.add_nodes_from(random_rows)
        bi_transitive_subgraph.add_nodes_from(random_cols)
        bi_transitive_subgraph.add_edges_from([( l,j) for j in random_cols for l in random_rows])
        rows=rows-random_rows
        cols=cols-random_cols

    # adding remaining nodes as singeltons
    if len(rows)>0:
        bi_transitive_subgraph.add_nodes_from(rows)
    if len(cols) > 0:
        bi_transitive_subgraph.add_nodes_from(cols)

    return bi_transitive_subgraph


def calculate_obj_val(weights, bitransgraph):
    obj_val=0
    num_rows = weights.shape[0]
    rows = [node for node in bitransgraph.nodes if helpers.is_row(node, num_rows)]
    cols = [node for node in bitransgraph.nodes if helpers.is_col(node, num_rows)]

    for i in rows:
        for j in cols:
            colID = helpers.node_to_col(j, num_rows)
            weight=weights[i][colID]
            # added edges
            if (i,j) in bitransgraph.edges and weight<0:
                obj_val += abs(weight)
            # deleted edges
            elif (i,j) not in bitransgraph.edges and weight>0 :
                obj_val += weight

    return obj_val


