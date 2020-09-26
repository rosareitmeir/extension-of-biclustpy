import networkx as nx
import movement


def run(weights, bi_transitive_subgrpah, obj_val, max_iter,nmin,nmax): # + max_iterations


    initialized_solution= movement.Solution(weights, bi_transitive_subgrpah)
    # local search for initialized solution
    best_solution,best_value= movement.execute_VND( obj_val,initialized_solution)
    best_iter=0
    cur_iter=0
    stopcond=False
    while not stopcond:
        shaked_solution,shaked_value=movement.shake_solution(nmin,nmax,best_solution,best_value,k=None)
        VND_solution,VND_value= movement.execute_VND(shaked_value,shaked_solution)
        # check acceptance criterion
        if VND_value < best_value:
                best_iter=cur_iter
                best_solution=VND_solution
                cur_iter += 1
        elif VND_value >= best_value and cur_iter- best_iter< max_iter+1:
                # bestsolution= bestsolution
                cur_iter += 1
                continue

        else :  # stop condition met
                #bestsolution= bestsolution
                stopcond=True



    # create optimized subgraph from best soultion bicluster set and return it

    optimized_subgraph= nx.subgraph
    return optimized_subgraph