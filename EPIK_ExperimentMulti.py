from EPIK_Utils import Property_Dist
from EPIK_Utils import save_solutions_all_csv
from EPIK_multi import EPIK_Multi
from enum import Enum
import os


class Algorithm (Enum):
    NSGAII = 0
    SPEA2  = 1
    CMAES  = 2


def run_experiment (data_dir, PROBLEM_NAME, RUNS, ALGORITHMS, NVARS, MODEL):
    #create directory
    dir_path = os.path.join(data_dir, PROBLEM_NAME)

    problem_dir         = os.path.join (data_dir, PROBLEM_NAME)
    paretoSet_dir       = os.path.join (problem_dir, "ParetoFrontSet")
    finalPopulation_dir = os.path.join (problem_dir, "FinalPopulation")

    os.makedirs(paretoSet_dir, exist_ok=True)
    os.makedirs(finalPopulation_dir, exist_ok=True)


    for algorithm in ALGORITHMS:
        print ("Running: ", algorithm.name)
        for run in range(RUNS):
            # initialise the EPIK instance
            epik_multi = EPIK_Multi(n, list_property_eval_func, list_property_eval_distr, list_property_prior_knowledge, MODEL)

            # execute the optimisation
            if algorithm is Algorithm.NSGAII:
                first_front, population, stats = epik_multi.run_NSGAI(BOUND_LOW, BOUND_UP, NVARS, POPULATION, GENERATIONS)
            elif algorithm is Algorithm.SPEA2:
                first_front, population, stats = epik_multi.run_SPEA2(BOUND_LOW, BOUND_UP, NVARS, POPULATION, GENERATIONS)
            elif algorithm is Algorithm.CMAES:
                first_front, population, stats = epik_multi.run_CMAES(BOUND_LOW, BOUND_UP, NVARS, POPULATION, GENERATIONS)

            suffix = algorithm.name +"-"+ str(run+1) + ".csv"
            # save Pareto front objectives and solutions to files
            #EPIK_Utils.save_objectives_csv(first_front, "ParetoFront" + suffix)
            #EPIK_Utils.save_solutions_csv(first_front, "ParetoSet" + suffix)

            # save population objectives and solutions to files
            #EPIK_Utils.save_objectives_csv(population, "FinalPopulationObjectives" + suffix)
            #EPIK_Utils.save_solutions_csv(population, "FinalPopulationSolutions" + suffix)

            # Save all
            save_solutions_all_csv(paretoSet_dir, "ParetoFrontSet"  + suffix, first_front)
            save_solutions_all_csv(finalPopulation_dir, "FinalPopulation" + suffix, population)


    from EPIK_Utils import find_reference_front_and_point
    substring = "ParetoFrontSet"
    reference_front_filepath, reference_point_filepath = find_reference_front_and_point(paretoSet_dir, substring, problem_dir)

    from EPIK_Indicators import calculate_indicators
    calculate_indicators(paretoSet_dir, substring, reference_front_filepath, reference_point_filepath, NVARS, problem_dir)




# if __name__ == '__main__':
#     # from ReqsPRESTO import *
#     #
#     # # Number of samples
#     # n = 100000
#     #
#     # # list keeping the formulae of properties for which we have prior knowledge
#     # list_property_eval_func = [property_R1_unknown_p1p2p3, property_R2_unknown_p1p2p3]
#     #
#     # # list keeping the prior knowledge for the properties of interest
#     # list_property_prior_knowledge = [dist_PK_R1b, dist_PK_R2b]
#     #
#     # # list keeping the type of properties for which we have prior knowledge
#     # # Property_Dist.Beta -> [0,1], Property_Dist.Gamma -> [0,\infty)
#     # list_property_eval_distr = [Property_Dist.Beta, Property_Dist.Gamma]
#     #
#     # assert len(list_property_eval_func) == len(list_property_prior_knowledge), "Inconsistent PK information"
#     # assert len(list_property_eval_func) == len(list_property_eval_distr), "Inconsistent PK information"
#     #
#     # NVARS =     6
#     #
#     #
#     # BOUND_LOW = 0.00001
#     # BOUND_UP = 1000
#     # POPULATION = 100
#     # GENERATIONS = 20
#     #
#     #
#     # ALGORITHMS = [Algorithm.CMAES, Algorithm.NSGAII, Algorithm.SPEA2]
#     # RUNS       = 5
#     # DATA_DIR   = "data"
#     # PROBLEM_NAME = "FPR_R1R2_p1p2p3b"
#     #
#     # run_experiment(DATA_DIR, PROBLEM_NAME, RUNS, ALGORITHMS, NVARS)



# if __name__ == '__main__':
#     from ReqsFX import *
#     from EPIK_Utils import MODEL_TYPE
#
#     # Number of samples
#     n = 100000
#
#     # list keeping the formulae of properties for which we have prior knowledge
#     # list_property_eval_func = [property_R1_unknown_p51, property_R2_unknown_p51]
#     list_property_eval_func = [property_R1_unknown_p41p51, property_R2_unknown_p41p51]
#     # list_property_eval_func = [property_R1_unknown_p21p41p51, property_R2_unknown_p21p41p51]
#
#     # list keeping the prior knowledge for the properties of interest
#     list_property_prior_knowledge = [dist_PK_R1b, dist_PK_R2b]
#
#     # list keeping the type of properties for which we have prior knowledge
#     # Property_Dist.Beta -> [0,1], Property_Dist.Gamma -> [0,\infty)
#     list_property_eval_distr = [Property_Dist.Beta, Property_Dist.Gamma]
#
#     assert len(list_property_eval_func) == len(list_property_prior_knowledge), "Inconsistent PK information"
#     assert len(list_property_eval_func) == len(list_property_eval_distr), "Inconsistent PK information"
#
#     NVARS =     2
#
#     MODEL = MODEL_TYPE.DTMC
#
#     BOUND_LOW = 0.0001
#     BOUND_UP = 10000
#     POPULATION = 100
#     GENERATIONS = 20
#
#
#     ALGORITHMS = [Algorithm.NSGAII, Algorithm.SPEA2, Algorithm.CMAES]
#     RUNS       = 30
#     DATA_DIR   = "data"
#     PROBLEM_NAME = "FX_R1R2_p5_100000"
#
#     run_experiment(DATA_DIR, PROBLEM_NAME, RUNS, ALGORITHMS, NVARS, MODEL)



if __name__ == '__main__':
    import ReqsDPM
    from EPIK_Utils import MODEL_TYPE
    import EPIK_Properties_Parser as Parser

    # Number of samples
    n = Parser.n

    # list keeping the formulae of properties for which we have prior knowledge
    list_property_eval_func = [ReqsDPM.property_R1_unknown_evTrans11, ReqsDPM.property_R4_unknown_evTrans11]

    # list keeping the prior knowledge for the properties of interest
    list_property_prior_knowledge = [ReqsDPM.dist_PK_R1, ReqsDPM.dist_PK_R2]

    # list keeping the type of properties for which we have prior knowledge
    # Property_Dist.Beta -> [0,1], Property_Dist.Gamma -> [0,\infty)
    list_property_eval_distr = [Property_Dist.Gamma, Property_Dist.Gamma]

    assert len(list_property_eval_func) == len(list_property_prior_knowledge), "Inconsistent PK information"
    assert len(list_property_eval_func) == len(list_property_eval_distr), "Inconsistent PK information"

    NVARS =     Parser.NVARS

    MODEL = Parser.MODEL_TYPE

    BOUND_LOW = 0.0001
    BOUND_UP = 10000
    POPULATION = Parser.POPULATION
    GENERATIONS = Parser.GENERATIONS


    ALGORITHMS = [Algorithm.NSGAII]
    RUNS       = Parser.RUNS
    DATA_DIR   = Parser.DATA_DIR

    # problem_name
    PROBLEM_NAME = Parser.PROBLEM_NAME + "_" + str(Parser.n)

    run_experiment(DATA_DIR, PROBLEM_NAME, RUNS, ALGORITHMS, NVARS, MODEL)



