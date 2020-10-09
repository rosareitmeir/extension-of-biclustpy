import copy
from collections import defaultdict
import numpy as np
import networkx as nx
import helpers

def execute_Rule2(graph):

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
        while len(R)>len(T):
            removed_node= np.random.choice(R)
            removed_nodes.append(removed_node)
            R.remove(removed_node)
            graph.remove_node(removed_node)
        if len( removed_nodes)>0: all_removed_nodes[R.pop()]=removed_nodes

    return all_removed_nodes


def execute_NewRule(graph,weights,num_rows):
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
            collapsed_column= np.sum(weights[:,col_idx],axis=1)
            collapsed_node = col_idx[0]
            #lostcol =copy.deepcopy(weights[:,collapsed_node])
            weights[:,collapsed_node]=collapsed_column
            removed_nodes[collapsed_node] = col_idx[1:]
            #weights=np.delete(weights,col_idx[1:],axis=1)
        else:
            collapsed_row=np.sum(weights[R,:],axis=0)
            #lostcol=copy.deepcopy(weights[collapsed_node])
            weights[collapsed_node]=collapsed_row
            removed_nodes[collapsed_node] = R[1:]
            #weights=np.delete(weights,R[1:],axis=0)

    return removed_nodes


def find_all_ciritical_independent_sets(graph):
    r = {node: tuple(graph.adj[node]) for node in graph.nodes}
    reverse = defaultdict(set)
    {reverse[v].add(k) for k, v in r.items()}
    critical_sets= {k:v for k,v in reverse.items() if len(v)>1}
    return critical_sets