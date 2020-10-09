
import main as bp
import numpy as np
import argparse as ap

def main():
    """Provides command line interface of biclustpy.
    """
    
    parser = ap.ArgumentParser(description="Compute bi-clusters using bi-cluster editing.")
    instance = parser.add_mutually_exclusive_group(required=True)
    instance.add_argument("--load", help="Load instance from .npy file.", metavar="input-file")
    instance.add_argument("--random", nargs=4, help="Randomly generate instance with num-rows rows and num-cols columns whose cells are of the form ((random value between 0 and 1) - threshold).", metavar=("num-rows", "num-cols", "threshold", "seed"))
    parser.add_argument("--save", help="Save bi-clusters as XML file.", metavar="output-file")
    parser.add_argument("--alg", default="ILP", help="Employed algorithm. Default = ILP.", choices=["ILP", "CH","GRASP"])
    parser.add_argument("--metaheu", help="Employed meatheuristics.", choices=["ILS", "GVNS"])
    parser.add_argument("--metaheu_options", nargs=3, type=int, default=[20, 2,10], help="Options for the metaheuristic ILS: maximum number of iterations to find an improved solution, minimal and maximal number of pertubations.", metavar=("max-iter", "nmin","nmax"))
    parser.add_argument("--grasp_options", nargs=3, type=float, default=[10, 0.7,None], help="Options for the algorithm GRASP: maximum number of iterations to find best solution, alpha ( between 0 and 1) to sort pairs out w.r.t to their g-values,seed for random choice.", metavar=("max-iter", "alpha","seed"))
    parser.add_argument("--ilp_options", nargs=2, type=int, default=[60, 0], help="Options for the algorithm ILP: time limit in second and flag that indicates whether model should be tuned before optimization.", metavar=("time-limit", "tune"))
    parser.add_argument("--preprocess", default="New Rule", help="preprocessing method: Rule 2 or default New Rule")
    args = parser.parse_args()
    
    weights = np.array(0)
    if args.load is not None:
        weights = np.load(args.load)
    
    if args.random is not None:
        np.random.seed(int(args.random[3]))
        num_rows = int(args.random[0])
        num_cols = int(args.random[1])
        threshold = float(args.random[2])
        instance = "random"
        weights = np.random.rand(num_rows, num_cols) - (threshold * np.ones((num_rows, num_cols)))

    preprocessing_method= args.preprocess[0]

    algorithm = bp.Algorithm()
    algorithm.algorithm_name = args.alg
    algorithm.ilp_time_limit = args.ilp_options[0]
    algorithm.ilp_tune = args.ilp_options[1]
    algorithm.max_iter=int(args.grasp_options[0])
    algorithm.grasp_alpha=args.grasp_options[1]
    algorithm.seed= args.grasp_options[2]


    # NEW ROSA running with metaheuristic or not
    if args.metaheu is not None:
        metaheuristic = bp.Algorithm()
        metaheuristic.algorithm_name = args.metaheu
        metaheuristic.max_iter=args.metaheu_options[0]
        metaheuristic.nmin=args.metaheu_options[1]
        metaheuristic.nmax = args.metaheu_options[2]
        bi_clusters, obj_val, is_optimal = bp.compute_bi_clusters(weights, preprocessing_method, algorithm, metaheuristic)
    else:
        bi_clusters, obj_val, is_optimal = bp.compute_bi_clusters(weights, preprocessing_method, algorithm)
    
    if args.save is not None:
        instance = ""
        if args.load is not None:
            instance = args.load
        if args.random is not None:
            instance = "random (threshold=" + args.random[2] + ", seed=" + args.random[3] + ")"
        bp.save_bi_clusters_as_xml(args.save, bi_clusters, obj_val, is_optimal, instance)


main()