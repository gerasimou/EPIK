import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow_probability
import random
from math import log, sqrt

from EPIK_Utils import Property_Dist, MODEL_TYPE, beta_from_moments, gamma_from_moments
import EPIK_Utils
import tensorflow as tf

tfd = tensorflow_probability.distributions

random.seed(64)




class EPIK_Multi:

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
            return tfd.Beta(tf.constant(fit_a, dtype=tf.float32), tf.constant(fit_b, dtype=tf.float32))
            # return stats.beta(a=fit_a, b=fit_b)
        elif prop_distr is Property_Dist.Gamma:
            fit_a, fit_b = gamma_from_moments(sample_mean, sample_variance)
            return tfd.Gamma(tf.constant(fit_a, dtype=tf.float32), tf.constant(fit_b, dtype=tf.float32))
            # return stats.gamma(scale=1 / fit_b, a=fit_a)


    # Return the distribution of y, given the distribution of x
    def kl_divergence_func(self, p, q):
        # calculate the kl divergence
        #print(p, q)
        return sum(p[i] * log(p[i] / q[i]) for i in range(len(q)))

    def kl_divergence_func2(self, p, q, prop_distr=Property_Dist.Beta):
        b1 = p
        b2 = q
        if prop_distr is Property_Dist.Beta:
            kl_b1_b2 = EPIK_Utils.kl_divergence_beta(b1, b2)
            b2_a = tf.convert_to_tensor(b2.concentration1)
            b2_b = tf.convert_to_tensor(b2.concentration0)
            # # print("test")
            # b1_a = p.kwds['a']
            # b1_b = p.kwds['b']
            # b2_a = q.kwds['a']
            # b2_b = q.kwds['b']
            # b1 = tfd.Beta(tf.constant(b1_a, dtype=tf.float64), tf.constant(b1_b, dtype=tf.float64))
            # b2 = tfd.Beta(tf.constant(b2_a, dtype=tf.float64), tf.constant(b2_b, dtype=tf.float64))
            #
            # b1b = stats.beta(a=b1_a, b=b1_b)
            # b2b = stats.beta(a=b2_a, b=b2_b)

        elif prop_distr is Property_Dist.Gamma:
            kl_b1_b2 = EPIK_Utils.kl_divergence_gamma(b1, b2)
            b2_a = tf.convert_to_tensor(b2.concentration)
            b2_b = tf.convert_to_tensor(b2.rate)
            # # print("ERROR")
            # b1_a = p.kwds['a']
            # b1_b = 1 / p.kwds['scale']
            # b2_a = q.kwds['a']
            # b2_b = 1 / q.kwds['scale']
            # b1 = tfd.Gamma(tf.constant(b1_a, dtype=tf.float64), tf.constant(b1_b, dtype=tf.float64))
            # b2 = tfd.Gamma(tf.constant(b2_a, dtype=tf.float64), tf.constant(b2_b, dtype=tf.float64))
            #
            # b1b = stats.gamma(scale=1/b1_b, a=b1_a)
            # b2b = stats.gamma(scale=1/b2_b, a=b2_a)


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
    def loss_func_multi(self, list_a, list_b):
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
        return kl_divergence_list


    def run_NSGAI(self, BOUND_LOW, BOUND_UP, NVARS, POPULATION, GENERATIONS):
        from deap import base
        from deap import creator
        from deap import tools

        CXPB = 0.9

        # create the fitness function structure, as many as the size of list_property_eval_func
        NOBJ = len(self.list_property_eval_func)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)

        # define the individual class
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # define the classes for the individual and population
        toolbox = base.Toolbox()
        toolbox.register("attr_float", EPIK_Utils.uniform, BOUND_LOW, BOUND_UP, NVARS)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # define the evaluation and reproduction features of the GA
        toolbox.register("evaluate", self.objective_eval_multi)  # , return_values_of=["F"])
        # toolbox.decorate("evaluate", tools.DeltaPenalty(self.check_feasibility, 100, pentaly_func))

        toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0 / NVARS)
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=30.0)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("selectFront", tools.sortNondominated)

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
            ind.fitness.values = fit

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
            offspring = tools.selTournamentDCD(population, len(population))
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
                ind.fitness.values = fit

            # select the next generation
            population = toolbox.select(population + offspring, POPULATION)

            # Compile statistics about the population
            record = stats.compile(population)
            logbook.record(gen=gen, evals=len(population), **record)
            #print(logbook.stream)

        first_front = toolbox.selectFront(population, len(population), first_front_only=True)
        return first_front[0], population, logbook



    def run_SPEA2(self, BOUND_LOW, BOUND_UP, NVARS, POPULATION, GENERATIONS):
        from deap import base
        from deap import creator
        from deap import tools

        CXPB = 0.9

        # create the fitness function structure, as many as the size of list_property_eval_func
        NOBJ = len(self.list_property_eval_func)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)

        # define the individual class
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # define the classes for the individual and population
        toolbox = base.Toolbox()
        toolbox.register("attr_float", EPIK_Utils.uniform, BOUND_LOW, BOUND_UP, NVARS)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # define the evaluation and reproduction features of the GA
        toolbox.register("evaluate", self.objective_eval_multi)  # , return_values_of=["F"])
        toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0 / NVARS)
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=30.0)
        toolbox.register("select", tools.selSPEA2)
        toolbox.register("selectFront", tools.sortNondominated)

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
            ind.fitness.values = fit

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
            offspring = tools.selection.selTournament(population, len(population), 2)
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
                ind.fitness.values = fit

            # select the next generation
            population = toolbox.select(population + offspring, POPULATION)

            # Compile statistics about the population
            record = stats.compile(population)
            logbook.record(gen=gen, evals=len(population), **record)
            #print(logbook.stream)

        first_front = toolbox.selectFront(population, len(population), first_front_only=True)
        return first_front[0], population, logbook



    def run_CMAES(self, BOUND_LOW, BOUND_UP, NVARS, POPULATION, GENERATIONS):
        from deap import base,  creator, tools, cma

        MIN_BOUND = np.full((NVARS,), BOUND_LOW) #np.zeros(NVARS)
        MAX_BOUND = np.full((NVARS,), BOUND_UP)
        EPS_BOUND = 2.e-5

        def distance(feasible_ind, original_ind):
            """A distance function to the feasibility region."""
            #return sum((f - o) ** 2 for f, o in zip(feasible_ind, original_ind))
            #return ( (feasible_ind[i] - original_ind[i]) ** 2 for i in range(len(feasible_ind.fitness.values)))
            #return sqrt(np.mean(np.square(np.subtract(feasible_ind.fitness.values, original_ind.fitness.values))))
            return sqrt(np.mean(np.square(np.subtract(feasible_ind, original_ind))))

        def closest_feasible(individual):
            """A function returning a valid individual from an invalid one."""
            feasible_ind = np.array(individual)
            feasible_ind = np.maximum(MIN_BOUND, feasible_ind)
            feasible_ind = np.minimum(MAX_BOUND, feasible_ind)
            return feasible_ind

        def valid(individual):
            """Determines if the individual is valid or not."""
            if any(individual < MIN_BOUND) or any(individual > MAX_BOUND):
                return False
            return True

        # def close_valid(individual):
        #     """Determines if the individual is close to valid."""
        #     if any(individual < MIN_BOUND - EPS_BOUND) or any(individual > MAX_BOUND + EPS_BOUND):
        #         return False
        #     return True

        # create the fitness function structure, as many as the size of list_property_eval_func
        NOBJ = len(self.list_property_eval_func)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)

        # define the individual class
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # define the classes for the individual and population
        toolbox = base.Toolbox()
        toolbox.register("attr_float", EPIK_Utils.uniform, BOUND_LOW, BOUND_UP, NVARS)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # define the evaluation and reproduction features of the GA
        toolbox.register("evaluate", self.objective_eval_multi)
        toolbox.decorate("evaluate", tools.ClosestValidPenalty(valid, closest_feasible, 1.0e+6, distance))
        toolbox.register("selectFront", tools.sortNondominated)

        # Initialize statistics object
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        # The MO-CMA-ES algorithm takes a full population as argument
        # population = [creator.Individual(x) for x in (np.random.uniform(0, 1, (MU, NVARS)))]
        population = toolbox.population(n=POPULATION)

        # evaluate the individuals and assign fitness value to each invividual from the population
        fitness_population = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitness_population):
            ind.fitness.values = fit

        MU, LAMBDA = POPULATION, POPULATION
        strategy = cma.StrategyMultiObjective(population, sigma=1.0, mu=MU, lambda_=LAMBDA)
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

        fitness_history = []

        # run the generation process
        print("Generation: ", end=" ")
        for gen in range(1, GENERATIONS):
            print(gen, " ", end=" ")

            # Generate a new population
            population = toolbox.generate()

            # evaluate the offspring and assign fitness values to each invividual
            fitness_population = toolbox.map(toolbox.evaluate, population)
            for ind, fit in zip(population, fitness_population):
                ind.fitness.values = fit
                fitness_history.append(fit)

            # Update the strategy with the evaluated individuals
            toolbox.update(population)

            # Compile statistics about the population
            record = stats.compile(population)
            logbook.record(gen=gen, evals=len(population), **record)
            #print(logbook.stream)

        first_front = toolbox.selectFront(population, len(population), first_front_only=True)
        return first_front[0], population, logbook



    def objective_eval_multi(self, x):
        list_a = [x[i] for i in range(len(x)) if i % 2 == 0]
        list_b = [x[i] for i in range(len(x)) if i % 2 == 1]

        return self.loss_func_multi(list_a, list_b)


    def check_feasibility (self, x):
        '''
            Feasibility function for the individual.
            Returns True if individual is feasible (or constraint not violated),
            False otherwise
        '''



    # Function to show the compared plot for the distribution of properties (with prior knowledge vs estimated)
    def showPlot (self, list_a, list_b):

        plots_num = len(self.list_property_eval_func)

        fig, axes = plt.subplots(1, plots_num, figsize=(plots_num * 5, 4))

        # Sample from the optimised dist. of parameters
        list_samples = [np.random.beta(list_a[i], list_b[i], size=n) for i in range(len(list_a))]

        for i in range(0, plots_num):
            # Sample from the PK
            samples_tar = self.list_property_prior_knowledge[i].rvs(n)

            # Calulate the sample of y via the sample of x
            sampled_Y = self.list_property_eval_func[i](list_samples)

            # Plot a histogram of the samples from the true and estimated distributions
            if (plots_num == 1):
                axis = axes
            else:
                axis = axes[i]
            sns.histplot(ax=axis, data=samples_tar, kde=True, label="PK")
            sns.histplot(ax=axis, data=sampled_Y, kde=True, label="Optimised")

        plt.show()



if __name__ == '__main__':
    from ReqsPRESTO import *

    # Number of samples
    n = 100000

    # list keeping the formulae of properties for which we have prior knowledge
    list_property_eval_func = [property_R1_unknown_p3,  property_R2_unknown_p3]

    # list keeping the type of properties for which we have prior knowledge
    # Property_Dist.Beta -> [0,1], Property_Dist.Gamma -> [0,\infty)
    list_property_eval_distr = [Property_Dist.Beta, Property_Dist.Gamma]

    # list keeping the prior knowledge for the properties of interest
    list_property_prior_knowledge = [dist_PK_R1b, dist_PK_R2b]

    model_type = MODEL_TYPE.DTMC

    # initialise the EPIK instance
    epik_multi = EPIK_Multi(n, list_property_eval_func, list_property_eval_distr, list_property_prior_knowledge, model_type)

    BOUND_LOW = 0.00001
    BOUND_UP = 1000
    NVARS = 2
    POPULATION = 40
    GENERATIONS = 40

    DATA_DIR = "data"

    # execute the optimisation
    # first_front, population, stats = epik_multi.run_NSGAI(BOUND_LOW, BOUND_UP, NVARS, POPULATION, GENERATIONS)
    # first_front, population, stats = epik_multi.run_CMAES(BOUND_LOW, BOUND_UP, NVARS, POPULATION, GENERATIONS)
    first_front, population, stats = epik_multi.run_SPEA2(BOUND_LOW, BOUND_UP, NVARS, POPULATION, GENERATIONS)

    # save Pareto front objectives and solutions to files
    #EPIK_Utils.save_objectives_csv(DATA_DIR, "ParetoFront.csv", first_front)
    # EPIK_Utils.save_solutions_csv(DATA_DIR,  "ParetoSet.csv",   first_front)

    # save population objectives and solutions to files
    # EPIK_Utils.save_objectives_csv(DATA_DIR, "FinalPopulationObjectives.csv", population)
    # EPIK_Utils.save_solutions_csv(DATA_DIR,  "FinalPopulationSolutions.csv",  population)

    # Save all
    EPIK_Utils.save_solutions_all_csv(DATA_DIR, "ParetoFrontSet.csv", first_front)
    EPIK_Utils.save_solutions_all_csv(DATA_DIR, "FinalPopulation.csv", population)

    # Save stats

    # show scatter plot
    #EPIK_Utils.show_scatterPlot_Front(first_front, population)


