import networkx as nx

import movement


def run(weights, bi_transitive_subgrpah, cur_val,nmin,nmax,maxiter):
    best_solution= movement.Solution(weights, bi_transitive_subgrpah)
    best_val=cur_val
    best_iter=0

    cur_iter=1
    while cur_iter<=maxiter: # stop condition ?
        k=1
        while k <3:
            # shake function ( solution obj , cur_val) creates new random solution saved in new solution obj
            shaked_sol,shaked_val = movement.shake_solution(nmin, nmax, best_solution, k)
            VND_sol, VND_val=movement.execute_VND(shaked_val, shaked_sol) # local search
            if VND_val < best_val:
                best_solution=VND_sol
                best_val=VND_val
                k=1

            else: k+=1
        #if
        #elif cur_val-best_iter <maxiter: k +=1
        #else: break # break while, no further improvment

    # matrix bicluster set to subgraph and return it
    optimized_subgraph= nx.subgraph
    return optimized_subgraph