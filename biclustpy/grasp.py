import numpy as np
import ch
import localsearch
import time
import helpers

def run(weights, subgrpah, maxiter,alpha,seed, timeout, queue=None):
   """ Greedy Randomzed Adaptve Search Procedure
   according to 4.6 (page 15)
   given weights of edges, a non bi-transitive subgraph, a number of maximal iterations to find a better solution and an alpha to select randomly pairs
   returns a solution which is generated randomly even so greedy and is optimized by local search
   """
   best_solution = None
   best_value = np.inf
   time_of_best=0
   best_iter = 0
   cur_iter = 0
   num_rows = weights.shape[0]
   if queue==None:
    queue= ch.calculate_g_values(subgrpah,weights,num_rows, alpha)
   start = time.time()
   elapsed = 0

   while cur_iter - best_iter <= maxiter and elapsed < timeout:
       # construction phase : create solution using CH run-method with alpha>1 -> random choice of pairs
       bi_transitive_subgrpah,value,optimal=ch.run(weights,subgrpah,alpha,seed,queue)
       cur_solution=localsearch.Solution(weights, bi_transitive_subgrpah)
       # local search phase: VND executed for obtained solution
       improved_solution, value= localsearch.execute_VND(value, cur_solution)
       if value < best_value:
            best_solution= improved_solution
            best_value=value
            time_of_best=time.time()-start
            best_iter=cur_iter
       cur_iter +=1
       elapsed= time.time()- start
   grasp_subgraph= helpers.graph_from_components(best_solution.bicluster_set)
   return grasp_subgraph, best_value, False, time_of_best





