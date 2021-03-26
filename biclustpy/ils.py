import helpers
import localsearch
import time

def run(weights, bi_transitive_subgrpah, obj_val, max_iter,nmin,nmax, timeout):
    '''Iterated Local Search
    according to 4.5 ( page 13)
    given weights of edges, intitialized bi-transtive graph and its objective value, minimal and maximal number of perturbations
    returns an optimized solution and its value
    '''

    initialized_solution= localsearch.Solution(weights, bi_transitive_subgrpah)
    start = time.time()
    time_to_best =0
    best_solution,best_value= localsearch.execute_VND(obj_val, initialized_solution)
    if best_value != obj_val:
        time_to_best = time.time()-start
    best_iter=0
    cur_iter=0
    stopcond=False


    elapsed = 0
    print(str(obj_val))
    while not stopcond and elapsed <timeout:
        shaked_solution,shaked_value=localsearch.shake_solution(nmin, nmax, best_solution, best_value, k=None)
        VND_solution,VND_value= localsearch.execute_VND(shaked_value, shaked_solution)
        # check acceptance criterion
        if VND_value < best_value:
                best_iter=cur_iter
                best_solution=VND_solution
                best_value=VND_value
                time_to_best=  time.time()-start
                print("hello")
                cur_iter += 1
        elif VND_value >= best_value and cur_iter- best_iter< max_iter+1:
                # bestsolution= bestsolution
                cur_iter += 1

        else :  # stop condition met
                #bestsolution= bestsolution, no better solution in maximum of iterations found
                stopcond=True

        elapsed= time.time()-start
    # create optimized subgraph from best solution bicluster set and return it
    optimized_subgraph= helpers.graph_from_components(best_solution.bicluster_set)
    return optimized_subgraph ,best_value, False, time_to_best