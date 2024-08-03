
from EPIK_single import EPIK_Single
from EPIK_Utils import Property_Dist, save_solution_all_single
import EPIK_Properties_Parser as Parser


# from ReqsPRESTO import *
# if __name__ == '__main__':
#
#     RUNS = Parser.RUNS
#
#     for RUN in range(RUNS):
#         #problem_name
#         problem_name = Parser.PROBLEM_NAME +"_"+ str(Parser.CONFORMANCE_LEVEL)
#
#         # Number of samples
#         n = Parser.n
#
#         BOUND_LOW = 0.00001
#         BOUND_UP = 1000
#         NVARS = Parser.NVARS
#         POPULATION = Parser.POPULATION
#         GENERATIONS = Parser.GENERATIONS
#
#         DATA_DIR = Parser.DATA_DIR
#
#         # list keeping the formulae of properties for which we have prior knowledge
#         list_property_eval_func = [property_R2_unknown_p3]
#
#         # list keeping the type of properties for which we have prior knowledge
#         # Property_Dist.Beta -> [0,1], Property_Dist.Gamma -> [0,\infty)
#         list_property_eval_distr = [Property_Dist.Gamma]
#
#         #model type
#         model_type = Parser.MODEL_TYPE
#
#
#         # list keeping the prior knowledge for the properties of interest
#         if Parser.CONFORMANCE_LEVEL == 0:
#             list_property_prior_knowledge = [dist_PK_R2]
#         elif Parser.CONFORMANCE_LEVEL == 1:
#             list_property_prior_knowledge = [dist_PK_R1, dist_PK_R2_conflict]
#         elif Parser.CONFORMANCE_LEVEL == 2:
#             list_property_prior_knowledge = [dist_PK_R1, dist_PK_R2_conflictb]
#         elif Parser.CONFORMANCE_LEVEL == 3:
#             list_property_prior_knowledge = [dist_PK_R1, dist_PK_R2_conflictC]
#         else:
#             raise Exception("Incorrect Conformance Level")
#
#         # initialise the EPIK instance
#         epik_single = EPIK_Single(n, list_property_eval_func, list_property_eval_distr, list_property_prior_knowledge, model_type)
#
#         # execute the optimisation
#         best_ind, population, stats = epik_single.run_singleGA(BOUND_LOW, BOUND_UP, NVARS, POPULATION, GENERATIONS)
#
#         #extract the details of the best solution
#         list_a = [best_ind[i] for i in range(len(best_ind)) if i % 2 == 0]
#         list_b = [best_ind[i] for i in range(len(best_ind)) if i % 2 == 1]
#
#
#         #save best solutions
#         bestSolutionsFilename = DATA_DIR +"/"+ problem_name +"_BestSolutions0.csv"
#         print(bestSolutionsFilename)
#         save_solution_all_single(bestSolutionsFilename, best_ind)
#
#
#         # #produce boxplot
#         # CONFORMANCE_OPTIONS = [0, 1, 2, 3]
#         # conformance_files = ['FPR_Conformance_' + str(i) + '_BestSolutions.csv' for i in CONFORMANCE_OPTIONS]
#         # conformanceLabels = ['Conformance', 'Low Conflict', 'Medium Conflict', 'High Conflict']
#         # colHeaders = [conformanceLabels[i] for i in CONFORMANCE_OPTIONS]
#         #
#         # from EPIK_visualisation import do_conformance_boxplots, show_distPlot2, do_conformance_distPlots
#         # # do_conformance_boxplots(data_dir=DATA_DIR, conformance_files=conformance_files, colHeaders=colHeaders)
#         #
#         # outFilename = DATA_DIR + "/" + problem_name + "_"
#         # property_labels = ["R1", "R2"]
#         # do_conformance_distPlots(data_dir=DATA_DIR, conformance_files=conformance_files, list_property_eval_func=list_property_eval_func,
#         #                          n=n, list_property_prior_knowledge=list_property_prior_knowledge, outFilename=outFilename, conformance_labels=CONFORMANCE_OPTIONS, property_labels=property_labels)






from ReqsDPM import  *
if __name__ == '__main__':

    RUNS = Parser.RUNS

    for RUN in range(RUNS):
        #problem_name
        problem_name = Parser.PROBLEM_NAME +"_"+ str(Parser.CONFORMANCE_LEVEL)

        # Number of samples
        n = Parser.n

        BOUND_LOW = 0.1
        BOUND_UP = 100
        NVARS = Parser.NVARS
        POPULATION = Parser.POPULATION
        GENERATIONS = Parser.GENERATIONS

        DATA_DIR = Parser.DATA_DIR

        # list keeping the formulae of properties for which we have prior knowledge
        list_property_eval_func = [property_R1_unknown_service]

        # list keeping the type of properties for which we have prior knowledge
        # Property_Dist.Beta -> [0,1], Property_Dist.Gamma -> [0,\infty)
        list_property_eval_distr = [Property_Dist.Gamma]

        #model type
        model_type = Parser.MODEL_TYPE


        # list keeping the prior knowledge for the properties of interest
        list_property_prior_knowledge = [dist_PK_R1_service]

        # initialise the EPIK instance
        epik_single = EPIK_Single(n, list_property_eval_func, list_property_eval_distr, list_property_prior_knowledge, model_type)

        # execute the optimisation
        best_ind, population, stats = epik_single.run_singleGA(BOUND_LOW, BOUND_UP, NVARS, POPULATION, GENERATIONS)

        #extract the details of the best solution
        list_a = [best_ind[i] for i in range(len(best_ind)) if i % 2 == 0]
        list_b = [best_ind[i] for i in range(len(best_ind)) if i % 2 == 1]


        #save best solutions
        bestSolutionsFilename = DATA_DIR +"/"+ problem_name +"_BestSolutions0.csv"
        print(bestSolutionsFilename)
        save_solution_all_single(bestSolutionsFilename, best_ind)