import numpy as np
import ch
import localsearch
import helpers

def run(weights, subgrpah, maxiter,alpha,seed):
   best_solution = None
   best_value = np.inf
   best_iter = 0
   cur_iter = 0
   while cur_iter - best_iter <= maxiter:
       # construct bigraph
       bi_transitive_subgrpah,value,optimal=ch.run(weights,subgrpah,alpha,seed) # alpha != 1 , random choice of pairs
       cur_solution=localsearch.Solution(weights, bi_transitive_subgrpah)
       # local search for obtained solution
       improved_solution, value= localsearch.execute_VND(value, cur_solution)
       if value < best_value:
            best_solution= improved_solution
            best_value=value
            best_iter=cur_iter
       cur_iter +=1


   grasp_subgraph= helpers.graph_from_components(best_solution.bicluster_set)
   return grasp_subgraph, best_value, False





