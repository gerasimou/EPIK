import numpy as np
import tensorflow_probability
import random
import tensorflow as tf

from EPIK_Utils import Property_Dist, MODEL_TYPE, beta_from_moments, gamma_from_moments
import EPIK_Utils

tfd = tensorflow_probability.distributions

random.seed(10)




class EPIK_Single:

    def __init__(self, samples_n, list_property_eval_func, list_property_eval_distr, list_property_prior_knowledge, model_type):
        self.n = samples_n
        self.list_property_eval_func = list_property_eval_func
        self.list_property_prior_knowledge = list_property_prior_knowledge
        self.list_property_eval_distr = list_property_eval_distr
        self.optimisation_iter = 0
        self.model_type = model_type


        self.KL_storage = []


    # Return the distribution of y, given the distribution of x
    def y_dist_from_x(self, list_a, list_b, n, property_eval_func, prop_distr=Property_Dist.Beta):
        # Inputs are the two Beta distribution parameters of x
        # list_a: the alpha parameters of Beta distribution using the traditional parameterisation for the parameters of interest
        # list_b: the beta parameter of Beta distribution using the traditional parameterisation for the parameters of interest
        # n: size of samples
        # property_eval_func: the property function to be evaluated (sampled)

        # Generate n samples from the beta distribution for the three parameters (as many as the size of the list)
        #TODO: A change here is needed to support CTMCs - DONE
        if self.model_type == MODEL_TYPE.DTMC.value:
            list_samples = [np.random.beta(list_a[i], list_b[i], size=n) for i in range(len(list_a))]
        elif self.model_type == MODEL_TYPE.CTMC.value:
            list_samples = [np.random.gamma(shape=list_a[i], scale=list_b[i], size=n) for i in range(len(list_a))]
        else:
            raise TypeError("Model type not DTMC or CTMC: " + self.model_type)

        # Sample for the property of interest given the generated values
        sampled_Y = property_eval_func(list_samples)

        # Calculate statistics (moments)
        sample_mean = np.mean(sampled_Y)
        sample_variance = np.var(sampled_Y)

        # Calculate distributions parameters from moments,
        # depending on its type and return the distributions
        if prop_distr is Property_Dist.Beta:
            fit_a, fit_b = beta_from_moments(sample_mean, sample_variance)
            return tfd.Beta(tf.constant(fit_a, dtype=tf.float64), tf.constant(fit_b, dtype=tf.float64))
            # return stats.beta(a=fit_a, b=fit_b)
        elif prop_distr is Property_Dist.Gamma:
            # fit_a, fit_b = gamma_from_moments(sample_mean, sample_variance)
            # return tfd.Gamma(tf.constant(fit_a, dtype=tf.float64), tf.constant(fit_b, dtype=tf.float64))

            # return stats.gamma(scale=1 / fit_b, a=fit_a)

            return tfd.Gamma.experimental_from_mean_variance(sample_mean, sample_variance)



    def kl_divergence_func2(self, p, q, prop_distr=Property_Dist.Beta):
        b1 = p
        b2 = q
        if prop_distr is Property_Dist.Beta:
            kl_b1_b2 = EPIK_Utils.kl_divergence_beta(b1, b2)
        #     # print("test")
        #     b1_a = p.kwds['a']
        #     b1_b = p.kwds['b']
        #     b2_a = q.kwds['a']
        #     b2_b = q.kwds['b']
        #     b1 = tfd.Beta(tf.constant(b1_a, dtype=tf.float64), tf.constant(b1_b, dtype=tf.float64))
        #     b2 = tfd.Beta(tf.constant(b2_a, dtype=tf.float64), tf.constant(b2_b, dtype=tf.float64))
            b2_a = tf.convert_to_tensor(b2.concentration1)
            b2_b = tf.convert_to_tensor(b2.concentration0)
        elif prop_distr is Property_Dist.Gamma:
            kl_b1_b2 = tfd.kl_divergence(b1, b2)#EPIK_Utils.kl_divergence_gamma(b1, b2)
        #     b1_a = p.kwds['a']
        #     b1_b = 1 / p.kwds['scale']
        #     b2_a = q.kwds['a']
        #     b2_b = 1 / q.kwds['scale']
        #     b1 = tfd.Gamma(tf.constant(b1_a, dtype=tf.float64), tf.constant(b1_b, dtype=tf.float64))
        #     b2 = tfd.Gamma(tf.constant(b2_a, dtype=tf.float64), tf.constant(b2_b, dtype=tf.float64))
            b2_a = tf.convert_to_tensor(b2.concentration)
            b2_b = tf.convert_to_tensor(b2.rate)


        # kl_b1_b2    = tfd.kl_divergence(b1, b2)
        kl_b1_b2_v = kl_b1_b2.numpy()
        if kl_b1_b2_v < 0 and kl_b1_b2_v >-0.001:
            kl_b1_b2_v = -kl_b1_b2_v

        msg = "KL negative - WRONG " + "KL(" + str(b2_a) +","+ str(b2_b) +") = " +  str(kl_b1_b2_v)
        try:
            assert kl_b1_b2_v >= 0, msg
        except AssertionError as e:
            print(e)
            exit(-1)

        return kl_b1_b2_v

    # KL({'scale': 1e-05, 'a': 2000000}, {'scale': 6.0300384427673324e-05, 'a': 330998.2364550653}) = -1.0

    # Loss function
    def loss_func_single(self, list_a, list_b):
        # list_a: the alpha parameters of Beta distribution using the traditional parameterisation for the parameters of interest
        # list_b: the beta parameter of Beta distribution using the traditional parameterisation for the parameters of interest
        # list_property_eval_func: list of the functions (properties) of interest

        # Calculate the KL divergence between the two distributions
        kl_divergence_list = []
        for i in range(len(self.list_property_eval_func)):
            sampled_distr = self.y_dist_from_x(list_a, list_b, self.n, self.list_property_eval_func[i],
                                               self.list_property_eval_distr[i])
            # KL divergence implementation
            kl_divergence = self.kl_divergence_func2(self.list_property_prior_knowledge[i], sampled_distr,
                                                     self.list_property_eval_distr[i])
            kl_divergence_list.append(kl_divergence)

        self.KL_storage.append(kl_divergence_list)
        return sum(kl_divergence_list)


    def run_singleGA(self, BOUND_LOW, BOUND_UP, NVARS, POPULATION, GENERATIONS):
        from deap import base
        from deap import creator
        from deap import tools

        CXPB = 0.9

        # create the fitness function structure, as many as the size of list_property_eval_func
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        # define the individual class
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # define the classes for the individual and population
        toolbox = base.Toolbox()
        toolbox.register("attr_float", EPIK_Utils.uniform, BOUND_LOW, BOUND_UP, NVARS)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # define the evaluation and reproduction features of the GA
        toolbox.register("evaluate", self.objective_eval_single)  # , return_values_of=["F"])
        # toolbox.decorate("evaluate", tools.DeltaPenalty(self.check_feasibility, 100, pentaly_func))

        toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0 / NVARS)
        # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=30.0)
        toolbox.register("mate", tools.cxTwoPoint)  # strategy for crossover, this classic two point crossover
        toolbox.register("select", tools.selTournament, tournsize=5)
        #toolbox.register("selectFront", tools.sortNondominated)

        # Initialize statistics object
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        population = toolbox.population(n=POPULATION)

        # evaluate the individuals and assign fitness value to each invividual from the population
        fitness_population = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitness_population):
            ind.fitness.values = [fit]

        # assign the crowding distance to the population
        population = toolbox.select(population, len(population))

        # Compile statistics about the population
        record = stats.compile(population)
        logbook.record(gen=0, evals=len(population), **record)
        #print(logbook.stream)

        # run the generation process
        print("Generation: ", end=" ")
        for gen in range(1, GENERATIONS):
            print(gen, " ", end=" ")
            # vary the population
            #offspring = tools.selTournamentDCD(population, len(population))
            offspring = toolbox.select(population, len(population))
            offspring = [toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= CXPB:
                    toolbox.mate(ind1, ind2)

                toolbox.mutate(ind1)
                toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values

            # evaluate the offspring and assign fitness values to each invividual
            fitness_offspring = toolbox.map(toolbox.evaluate, offspring)
            for ind, fit in zip(offspring, fitness_offspring):
                ind.fitness.values = [fit]

            # select the next generation
            population = toolbox.select(population + offspring, POPULATION)

            # Compile statistics about the population
            record = stats.compile(population)
            logbook.record(gen=gen, evals=len(population), **record)
            #print(logbook.stream)

        best_ind = tools.selBest(population, 1)[0]
        return best_ind, population, logbook



    def objective_eval_single(self, x):
        list_a = [x[i] for i in range(len(x)) if i % 2 == 0]
        list_b = [x[i] for i in range(len(x)) if i % 2 == 1]

        return self.loss_func_single(list_a, list_b)


    def check_feasibility (self, x):
        '''
            Feasibility function for the individual.
            Returns True if individual is feasible (or constraint not violated),
            False otherwise
        '''




if __name__ == '__main__':
    import EPIK_Properties_Parser as Parser
    from ReqsPRESTO import *

    # problem_name
    problem_name = Parser.PROBLEM_NAME + "_" + str(Parser.CONFORMANCE_LEVEL)

    # Number of samples
    n = Parser.n

    BOUND_LOW = 0.00001
    BOUND_UP = 100
    NVARS = Parser.NVARS
    POPULATION = Parser.POPULATION
    GENERATIONS = Parser.GENERATIONS

    DATA_DIR = Parser.DATA_DIR

    # list keeping the formulae of properties for which we have prior knowledge
    list_property_eval_func = [property_R1_unknown_p3, property_R2_unknown_p3]

    # list keeping the type of properties for which we have prior knowledge
    # Property_Dist.Beta -> [0,1], Property_Dist.Gamma -> [0,\infty)
    list_property_eval_distr = [Property_Dist.Beta, Property_Dist.Gamma]

    # list keeping the prior knowledge for the properties of interest
    if Parser.CONFORMANCE_LEVEL == 0:
        list_property_prior_knowledge = [dist_PK_R1, dist_PK_R2]
    elif Parser.CONFORMANCE_LEVEL == 1:
        list_property_prior_knowledge = [dist_PK_R1, dist_PK_R2_conflict]
    elif Parser.CONFORMANCE_LEVEL == 2:
        list_property_prior_knowledge = [dist_PK_R1, dist_PK_R2_conflictb]
    elif Parser.CONFORMANCE_LEVEL == 3:
        list_property_prior_knowledge = [dist_PK_R1, dist_PK_R2_conflictC]
    else:
        raise Exception("Incorrect Conformance Level")

    # initialise the EPIK instance
    epik_single = EPIK_Single(n, list_property_eval_func, list_property_eval_distr, list_property_prior_knowledge)

    # execute the optimisation
    best_ind, population, stats = epik_single.run_singleGA(BOUND_LOW, BOUND_UP, NVARS, POPULATION, GENERATIONS)

    print("\nFitness:\t", best_ind.fitness.values)
    print("Values:\t",  best_ind)
    print("---DONE---")

    from EPIK_visualisation import show_distPlot2, show_distPlot_TFD
    problem_name = "FPR_Conformance"
    list_a = [best_ind[i] for i in range(len(best_ind)) if i % 2 == 0]
    list_b = [best_ind[i] for i in range(len(best_ind)) if i % 2 == 1]



    outFilename = DATA_DIR +"/"+ problem_name +"_"
    labels = ["R1", "R2"]
    title = problem_name

    # show_distPlot_TFD(list_a, list_b, list_property_eval_func, n, list_property_prior_knowledge,
    #                outFilename, labels, title)

    from EPIK_Utils import save_solution_all_single

    #Save best solutions
    bestSolutionsFilename = outFilename +"BestSolutions.csv"
    save_solution_all_single(bestSolutionsFilename, best_ind)

    #Save KL history
    from EPIK_Utils import save_KL_history
    filename_KL = outFilename + "C" + str(Parser.CONFORMANCE_LEVEL) +"_KL.csv"
    save_KL_history(DATA_DIR, filename_KL, epik_single.KL_storage)
