
#from .
import main as bp
import numpy as np
import argparse as ap
import helpers as help
import csv
def main():
    """Provides command line interface of biclustpy.
    """
    
    parser = ap.ArgumentParser(description="Compute bi-clusters using bi-cluster editing.")
    instance = parser.add_mutually_exclusive_group(required=True)
    instance.add_argument("--load", help="Load instance from .npy file.", metavar="input-file")
    instance.add_argument("--random", nargs=4, help="Randomly generate instance with num-rows rows and num-cols columns whose cells are of the form ((random value between 0 and 1) - threshold).", metavar=("num-rows", "num-cols", "threshold", "seed"))
    parser.add_argument("--names", default=False, action="store_true", help="column and row names for instance is given.")
    parser.add_argument("--save", help="Save bi-clusters as XML file.", metavar="output-file")
    parser.add_argument("--alg", default="ILP", help="Employed algorithm. Default = ILP.", choices=["ILP", "CH","GRASP", "RANDOM"])
    parser.add_argument("--metaheu", help="Employed meatheuristics.", choices=["ILS", "GVNS"])
    parser.add_argument("--metaheu_options", nargs=4, type=str, default=[20, 2,10, "inf"], help="Options for the metaheuristic ILS and GVNS: maximum number of iterations to find an improved solution, minimal and maximal number of pertubations, time limit in sec.", metavar=("max-iter", "nmin","nmax", "time limit"))
    parser.add_argument("--grasp_options", nargs=4, type=str, default=[30, 0.5,"None", "inf"], help="Options for the algorithm GRASP: maximum number of iterations to find best solution, alpha ( between 0 and 1) to sort pairs out w.r.t to their g-values,seed for random choice.", metavar=("max-iter", "alpha","seed", "time limit"))
    parser.add_argument("--ilp_options", nargs=2, type=int, default=[60, 0], help="Options for the algorithm ILP: time limit in second and flag that indicates whether model should be tuned before optimization.", metavar=("time-limit", "tune"))
    parser.add_argument("--random_options", type=int, default=20, help="number of random initialization")
    parser.add_argument("--preprocess", type=str, nargs=2, default=["New", "Rule"], help="preprocessing method: Rule 2 or default New Rule")
    parser.add_argument("--calc_gvalues", default="", type=str, help="calculate and save gvalues, path to gvalue file")
    parser.add_argument("--load_gvalues",  type=str, default=None,  help="Option for CH/GRASP: loading g-values ", metavar=("path to gvalue file"))

    args = parser.parse_args()
    
    weights = np.array(0)
    names=None
    if args.load is not None:
        if args.load.endswith(".npy"):
            weights = np.load(args.load)
        elif args.load.endswith(".tsv"):
            if args.names:
                with open(args.load) as f:
                    reader = csv.reader(f, delimiter="\t")
                    columns = next(reader)
                    rownames=[]
                    for row in reader:
                        rownames.append(row[0])
                    numrow= len(rownames)
                    numcol= len(columns)
                    names = dict(zip(range(numrow, numrow+ numcol), columns))
                    names.update(dict(zip(range(0, len(rownames)), rownames)))

                weights= np.loadtxt(args.load, delimiter="\t", skiprows=1,usecols=range(1,numcol))

            else:  weights= np.loadtxt(args.load, delimiter="\t")
            weights = weights.astype(np.int)

                
        elif args.load.endswith(".csv"):
            weights= np.loadtxt(args.load, delimiter=",")

    
    if args.random is not None:
        np.random.seed(int(args.random[3]))
        num_rows = int(args.random[0])
        num_cols = int(args.random[1])
        threshold = float(args.random[2])
        instance = "random"
        weights = np.random.rand(num_rows, num_cols) - (threshold * np.ones((num_rows, num_cols)))

    preprocessing_method= args.preprocess[0] + " " + args.preprocess[1]

    algorithm = bp.Algorithm()
    algorithm.algorithm_name = args.alg
    algorithm.ilp_time_limit = args.ilp_options[0]
    algorithm.num_init=args.random_options
    algorithm.ilp_tune = args.ilp_options[1]
    algorithm.max_iter=int(args.grasp_options[0])
    algorithm.grasp_alpha=float(args.grasp_options[1])
    if args.grasp_options[2] == "None":
            algorithm.seed= None
    else:
        algorithm.seed=int(args.grasp_options[2])

    if args.grasp_options[3] == "inf":
        algorithm.grasp_time_limit= np.inf
    else:
        algorithm.grasp_time_limit = int(args.grasp_options[3])

    # reading g-value file
    all_gvalues=[]
    if args.load_gvalues != None:
        if names!= None:
            inv_names= {v:k for k, v in names.items()}
        file=open(args.load_gvalues, "r")
        cur_gv_list=None
        for line in file:
            if line.startswith("#") or line.startswith("g-values"):
                if cur_gv_list!=None:
                    all_gvalues.append(cur_gv_list)
                cur_gv_list = []
                continue
            split=line.split("\t")
            if names != None:
                row=inv_names[split[0]]
                col=inv_names[split[1]]
            else:
                row = int(split[0])
                col = int(split[1])
            gval=float(split[2])
            cur_gv_list.append(( (row, col), gval))
        file.close()
        all_gvalues.append(cur_gv_list)
        algorithm.gval= all_gvalues



    # initialize metaheuristic
    if args.metaheu is not None:
        metaheuristic = bp.Algorithm()
        metaheuristic.algorithm_name = args.metaheu
        metaheuristic.max_iter=int(args.metaheu_options[0])
        metaheuristic.nmin = int(args.metaheu_options[1])
        metaheuristic.nmax = int(args.metaheu_options[2])
        if args.metaheu_options[3]  != "inf":
            metaheuristic.meta_time_limit = int(args.metaheu_options[3])
        else:
            metaheuristic.meta_time_limit = np.inf

        bi_clusters, obj_val, is_optimal , time, metaheu_times = bp.compute_bi_clusters(weights, preprocessing_method, algorithm, False, metaheuristic)
    elif args.calc_gvalues=="":
        bi_clusters, obj_val, is_optimal, time, metaheu_times = bp.compute_bi_clusters(weights, preprocessing_method, algorithm, False)
    else:
        # calc g-values
        all_gvalues, times= bp.compute_bi_clusters(weights, preprocessing_method, algorithm, calc_gv=True)
        help.write_gvalue_list(args.calc_gvalues, all_gvalues, times,  names)

    
    if args.save is not None:
        instance = ""
        if args.load is not None:
            instance = args.load

        if args.random is not None:
            instance = "random (threshold=" + args.random[2] + ", seed=" + args.random[3] + ")"
        if len(metaheu_times)!=None:
            bp.save_bi_clusters_as_xml(args.save, bi_clusters, obj_val, is_optimal,time, instance, names, times=metaheu_times)
        else:
            bp.save_bi_clusters_as_xml(args.save, bi_clusters, obj_val, is_optimal,time, instance, names)


main()