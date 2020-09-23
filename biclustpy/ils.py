import networkx as nx
import movement


def run(weights, bi_transitive_subgrpah, obj_val, max_iterations):
    current_iteration=0
    iteration_with_best_solution=0
    initialized_solution= movement.Solution(weights, bi_transitive_subgrpah)

    best_solution,best_value= movement.execute_VND( obj_val,initialized_solution)

    #while i<max_iterations and bestsolution_found==False:
        # current_iteration+=1
        # shake solution (best_solution), return new solution
        # execute_VND(shaked solution) return optimized solution
        # check acceptance criterion
            # if value of VND solution < bestvalue
                #iteration_with_best_solution=current_iteration
                # best_solution=VND solution
            # elif value of VND solution >= bestvalue and current_iteration- iteration_with_best_soultion< max_iterations
                # nothing to change continue
            # else : best solution value and VNS solution have same obj_val
                #bestsoultion_found =True


    # create optimized subgraph from best soultion bicluster set and return it

    optimized_subgraph= nx.subgraph
    return optimized_subgraph