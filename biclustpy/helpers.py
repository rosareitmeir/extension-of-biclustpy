import networkx as nx
import xml.etree.ElementTree as ET
from xml.dom import minidom

import localsearch


def prettify(elem):
    """Returns a pretty-printed XML string for the ElementTree element.
    
    Args:
        elem (xml.etree.ElementTree.Element): The element that should be prettified.
    
    Returns:
        string: A pretty-printed XML string for the element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")
    
def build_element_tree(bi_clusters, obj_val, is_optimal, time, instance, names=None, times=None):
    root = ET.Element("bi_clusters")
    root.set("num_bi_clusters", str(len(bi_clusters)))
    root.set("num_rows", str(sum([len(bi_cluster[0]) for bi_cluster in bi_clusters])))
    root.set("num_cols", str(sum([len(bi_cluster[1]) for bi_cluster in bi_clusters])))
    root.set("obj_val", str(obj_val))
    root.set("is_opt", str(is_optimal))
    root.set("time", str(time))
    root.set("instance", instance)
    if times != None:
        root.set("time_till_optimization", ' '.join(map(str, times)))
    cluster_id = 0
    for bi_cluster in bi_clusters:
        child = ET.SubElement(root, "bi_cluster")
        child.set("id", "_" + str(cluster_id))
        child.set("num_rows", str(len(bi_cluster[0])))
        child.set("num_cols", str(len(bi_cluster[1])))

        rows = ET.SubElement(child, "rows")
        columns = ET.SubElement(child, "cols")
        if names != None:
            rows.text = " ".join([names[row] for row in bi_cluster[0]])
            columns.text = " ".join([names[col] for col in bi_cluster[1]])
        else:
            rows.text = " ".join([str(row+1) for row in bi_cluster[0]])
            columns.text = " ".join([str(col+1) for col in bi_cluster[1]])
        cluster_id = cluster_id + 1
    return root

def write_gvalue_list(path, gvalues, time, names=None):
    file=open(path, "w+")
    n=1
    file.write("#g-values total time "+ str(time)+ "\n")
    for subgraph in gvalues:
        file.write("#g-values for subgraph %d \r\n" % n)
        n+=1
        for ((i, k), g) in subgraph:
            if names==None:
                file.write(str(i)+ "\t"+ str(k)+ "\t"+ str(g)+ "\n")
            else:
                file.write(names[i]+ "\t"+ names[k]+ "\t"+ str(g)+ "\n")
    file.close()


def find_matching_gvalues(nodes, gvalues):
    for list in gvalues:
        i=list[0][0][0]
        if i in nodes:
            return list




def col_to_node(col, num_rows):
    """Returns node ID of a column in the instance.
    
    Args:
        col (int): The column ID.
        num_rows (int): The number of rows in the instance.
    
    Returns:
        int: The node ID of the column col.
    """
    return num_rows + col

def node_to_col(node, num_rows):
    """Returns column ID of a node in the instance's graph representation that represents a column.
    
    Args:
        node (int): The node ID of the column in the graph representation.
        num_rows (int): The number of rows in the instance.
    
    Returns:
        int: Column ID of the node. If negative, the node is not a column but a row.
    """
    return node - num_rows

def is_row(node, num_rows):
    """Checks if a node in the graph representation of the instance is a row.
    
    Args:
        node (int): The node ID in the graph representation.
        num_rows (int): The number of rows in the instance.
    
    Returns:
        bool: True if and only if node is a row.
    """
    return node < num_rows

def is_col(node, num_rows):
    """Checks if a node in the graph representation of the instance is a column.
    
    Args:
        node (int): The node ID in the graph representation.
        num_rows (int): The number of rows in the instance.
    
    Returns:
        bool: True if and only if node is a column.
    """
    return node >= num_rows
    
def build_graph_from_weights(weights, nodes):
    """Builds NetworkX graph for given weights, threshold, and subset of nodes.
    
    Args:
        weights (numpy.array): The overall problem instance.
        nodes (list of int): The set of nodes for which the graph should be build. 
            Must be a subset of range(weights.shape[0] + weights.shape[1]).
    
    Returns:
        networkx.Graph: The induced subgraph in the specified set of nodes.
    """
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    num_rows = weights.shape[0]
    for row in nodes:
        if is_col(row, num_rows):
            continue
        for col in nodes:
            if is_row(col, num_rows):
                continue
            if weights[row, node_to_col(col, num_rows)] > 0:
                graph.add_edge(row, col)
    return graph

def connected_components(graph):
    """Decomposes graph into connected components.
    
    Args:
        graph (networkx.Graph): Graph that should be decomposed.
    
    Returns:
        list of networkx.Graph: List of connected components.
    """
    nodes_of_components = list(nx.connected_components(graph))
    num_components = len(nodes_of_components)
    components = [nx.Graph() for i in range(num_components)]
    labels = {}
    for i in range(num_components):

        components[i].add_nodes_from(nodes_of_components[i])
        for node in nodes_of_components[i]:
            labels[node] = i
    for (node_1, node_2) in graph.edges:
        i = labels[node_1]
        if labels[node_2] == i:
            components[i].add_edge(node_1, node_2)
    return components
    
def is_bi_clique(graph, num_rows):
    """Checks if a bipartite graph is a bi-clique.
    
    Args:
        graph (networkx.Graph): Bipartite graph.
        num_rows (int): The number of nodes in the original instance. 
            Nodes are in the left partition of graph if and only if their ID is smaller than num_rows.
    
    Returns:
         bool: True if and only if the graph is a bi-clique.
    """
    size_left = 0
    size_right = 0
    for node in graph.nodes:
        if is_row(node, num_rows):
            size_left = size_left + 1
        else:
            size_right = size_right + 1
    return graph.number_of_edges() == size_left * size_right
    
def is_singleton(bi_cluster):
    ''' returns true if and only if there is one node in the given bicluster'''
    return (len(bi_cluster[0]) == 0) or (len(bi_cluster[1]) == 0)

# new Rosa
def build_bicluster(nodes1,nodes2=None):
    ''' returns a bi-transitive graph from nodes'''
    graph = nx.Graph()
    graph.add_nodes_from(nodes1)
    if nodes2!=None:
        graph.add_nodes_from(nodes2)
    V1_nodes= [elem for elem in graph.nodes if is_row(elem,localsearch.num_rows)]
    V2_nodes= [elem for elem in graph.nodes if is_col(elem,localsearch.num_rows)]
    graph.add_edges_from((x,y) for x in V1_nodes for y in V2_nodes)
    return graph


def graph_from_components(bicluster_set):
    graph= nx.Graph()
    for bicluster in bicluster_set:
        graph.add_nodes_from(bicluster.nodes)
        graph.add_edges_from(bicluster.edges)

    return graph

# for moving vertex
def is_singleton_in_same_partion(biclust1,partion):
    if len(biclust1.nodes)==1 and  list(biclust1.nodes)[0] in partion:
            return True
    return False

# for join bicluster
def are_singeltons_in_same_partion(biclust1,biclust2):
    if len(biclust1.nodes)==len(biclust2.nodes)==1 and is_row(list(biclust1.nodes)[0], localsearch.num_rows)==is_row(list(biclust2.nodes)[0], localsearch.num_rows):
           return True
    return False

# for break bicluster
def is_no_valid_biclique(biclust1, biclust2):
    '''returns true if and only if all nodes in the cluster are in the same partition'''
    if len(biclust1) > 1 and (all(is_row(x, localsearch.num_rows) for x in biclust1) or all(is_col(x, localsearch.num_rows) for x in biclust1)):
        return True
    if len(biclust2) > 1 and (all(is_row(x, localsearch.num_rows) for x in biclust2) or all(is_col(x, localsearch.num_rows) for x in biclust2)):
        return True
    return False


def find_possible_biclusters(bicluster_set,number_biclusters, num_rows):
    pos_after_cluster_V1 = list(range(number_biclusters))
    pos_after_cluster_V2 = list(range(number_biclusters))
    for i in range(number_biclusters):
        nodes=list(bicluster_set[i].nodes)
        if len(nodes) == 1:
            if is_row(nodes[0], num_rows):
                pos_after_cluster_V1.remove(i)
            else:
                pos_after_cluster_V2.remove(i)

    return pos_after_cluster_V1, pos_after_cluster_V2

def sort_out_singeltons(bicluster_set,number_biclusters):
    not_singeltons=[]

    for i in range(number_biclusters):
        nodes = list(bicluster_set[i].nodes)
        if len(nodes) > 1:
                not_singeltons.append(i)

    return not_singeltons



def is_better_than_cur_sol(newsol, newval, cursol, curval):
    if curval -newval >= 1:
        return True
    # weighted case
    if curval - newval >= 0.000001:
        #to be sure check if the sol are not the same
        if are_same_sol(newsol.bicluster_set, cursol.bicluster_set):
            return False
        else:
            return True

    return False


def are_same_sol(sol1, sol2):

    pos_matches= list(range(len(sol2)))
    for bic1 in sol1:
        match =False
        for i in pos_matches:
            bic2=sol2[i]
            if set(bic1.nodes) == set(bic2.nodes):
                match=True
                pos_matches.remove(i)
                break
        if not match:
            return False
    return True






