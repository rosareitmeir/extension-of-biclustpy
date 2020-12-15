import helpers
import localsearch
import time

def run(weights, bi_transitive_subgrpah, cur_val,maxiter,nmin,nmax, timeout):
    '''General Variable Neighbourhood Search
    according to 4.4 (page 12)
    given weights of the edges, an initialized solution with its objective value, maximal number of iterations until finding a better solution, minimal and maximal number of perturbations
    returns an optimized solution and its value
    '''
    best_solution= localsearch.Solution(weights, bi_transitive_subgrpah)
    best_val=cur_val
    best_iter=0
    cur_iter=0
    start = time.time()
    elapsed = 0
    while cur_iter-best_iter<=maxiter and elapsed < timeout: # allowed iterations until finding a better solution
        k=0
        better_sol_found=False
        while k <3:
            # shake current solution to escape local optima with fixed neighbourhood
            shaked_sol,shaked_val = localsearch.shake_solution(nmin, nmax, best_solution, best_val,k)
            # descent phase : local search using VND
            VND_sol, VND_val=localsearch.execute_VND(shaked_val, shaked_sol)
            if VND_val < best_val:
                best_solution=VND_sol
                best_val=VND_val
                better_sol_found=True
                k=1
            else: k+=1
        if better_sol_found:
            best_iter=cur_iter
        cur_iter +=1
        elapsed= time.time() -start

    optimized_subgraph= helpers.graph_from_components(best_solution.bicluster_set)
    return optimized_subgraph,best_val,False