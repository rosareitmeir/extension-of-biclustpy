import helpers
import movement


def run(weights, bi_transitive_subgrpah, cur_val,maxiter,nmin,nmax):
    best_solution= movement.Solution(weights, bi_transitive_subgrpah)
    best_val=cur_val
    best_iter=0
    cur_iter=0
    while cur_iter-best_iter<=maxiter: # stop condition
        k=1
        better_sol_found=False
        while k <3:
            # shake function ( solution obj , cur_val) creates new random solution saved in new solution obj
            shaked_sol,shaked_val = movement.shake_solution(nmin, nmax, best_solution, k)
            VND_sol, VND_val=movement.execute_VND(shaked_val, shaked_sol) # local search
            if VND_val < best_val:
                best_solution=VND_sol
                best_val=VND_val
                better_sol_found=True
                k=1
            else: k+=1
        if better_sol_found:
            best_iter=cur_iter
        cur_iter +=1

    # matrix bicluster set to subgraph and return it
    optimized_subgraph= helpers.graph_from_components(best_solution.bicluster_set)
    return optimized_subgraph,best_val,False