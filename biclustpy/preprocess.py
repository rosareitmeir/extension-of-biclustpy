
from collections import defaultdict
import numpy as np
import helpers

def execute_Rule2(graph):
    # according to  Rule 2 on page 7
    critical_independent_sets= find_all_ciritical_independent_sets(graph)
    all_removed_nodes={}
    for S,R in critical_independent_sets.items():
        # R is a critical independent set
        # S is the open vertex neighbourhood of R
        # define T = N(S)\R : the neighbourhood of the nodes, which are in the neighbourhood of the current considering critical set R
        T=set()
        for adjnode in S:
            T.update(tuple(graph[adjnode]))
        T=(T-R)
        removed_nodes=[]
        while len(R)>len(T): # till |R|=|T|
            removed_node= np.random.choice(R)
            removed_nodes.append(removed_node)
            R.remove(removed_node)
            graph.remove_node(removed_node)
        if len( removed_nodes)>0: all_removed_nodes[R.pop()]=removed_nodes
    return all_removed_nodes


def execute_NewRule(graph,weights,num_rows):
    # according to New Rule definition on page 7
    critical_independent_sets= find_all_ciritical_independent_sets(graph)
    removed_nodes={}
    for R in critical_independent_sets.values():
        R=list(R)
        # keep one node to summarize them all to one
        collapsed_node = R[0]
        graph.remove_nodes_from(R[1:])
        # editing weights matrix
        # collapsing parallel edges to one edge
        if helpers.is_col(collapsed_node, num_rows):
            col_idx=[helpers.node_to_col(node,num_rows) for node in R]
            collapsed_node_idx = col_idx[0]
            # create accumulated edges
            collapsed_column= np.sum(weights[:,col_idx],axis=1)
            weights[:,collapsed_node_idx]=collapsed_column
            removed_nodes[collapsed_node] = R[1:]
        else:
            # create accumulated edges
            collapsed_row=np.sum(weights[R,:],axis=0)
            weights[collapsed_node]=collapsed_row
            removed_nodes[collapsed_node] = R[1:]

    return removed_nodes


def find_all_ciritical_independent_sets(graph):
    #Defintion of critical independent set is according to page 6 Defintion 1
    # dictionary with node as kex and its open vertex neighbourhood as value
    # invert it and keep sets with at least two nodes, which are sharing the same vertex neighbourhood.
    r = {node: tuple(graph.adj[node]) for node in graph.nodes}
    reverse = defaultdict(set)
    {reverse[v].add(k) for k, v in r.items()}
    critical_sets= {k:v for k,v in reverse.items() if len(v)>1}
    return critical_sets