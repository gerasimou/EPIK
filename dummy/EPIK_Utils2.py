import seaborn as sns
import matplotlib.pyplot as plt
import csv
import pandas as pd
import random
import numpy as np
from enum import Enum
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import os


class Property_Dist (Enum):
    Beta = 0
    Gamma = 1


# Toolbox initialization
def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]



#Function to calculate the two Beta parameters given the mean and variance of the data
def beta_from_moments(m, v):
    # m: mean
    # v: variance
    return m*(m-v-m**2)/(v), (m-v-m**2)*(1-m)/(v)



#Function to calculate the two Gamma parameters given the mean and variance of the data
def gamma_from_moments(m, v):
  # m: mean
  # v: variance
  return (m**2)/v, m/v


def save_objectives_csv(data_dir, filename, solutions, ):
    # Find the number of columns
    colsNum = len(solutions[0].fitness.values)

    # Create the column headers
    colHeaders = ["P" + str(i + 1) for i in range(colsNum)]

    # Create value lists
    objective_values = [sol.fitness.values for sol in solutions]

    filepath = os.path.join(data_dir, filename)

    with open(filepath, 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write the header
        writer.writerow(colHeaders)

        # write multiple rows
        writer.writerows(objective_values)

    print(filepath + " saved")



def save_solutions_csv(data_dir, filename, solutions):
    # Find the number of solution items
    solItems = len(solutions[0])

    # Create solutions lists
    solution_values = [[sol[i] for i in range(solItems)] for sol in solutions]

    filepath = os.path.join(data_dir, filename)

    with open(filepath, 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write the header
        #writer.writerow(colHeaders)

        # write multiple rows
        writer.writerows(solution_values)

    print(filepath + " saved")


def save_solutions_all_csv(data_dir, filename, solutions):
    # Find the number of properties
    objsNum = len(solutions[0].fitness.values)

    # Find the number of solution items
    solItems = len(solutions[0])

    # Create data to save
    data = [[sol[i] for i in range(solItems)] for sol in solutions]

    # Add the objective values
    for i in range(len(solutions)):
        data[i].extend(list(solutions[i].fitness.values))

    # Create the column headers
    headers = ["v" + str(i + 1) for i in range(solItems)]
    colHeaders = ["P" + str(i + 1) for i in range(objsNum)]
    headers.extend(colHeaders)
    #print(headers)

    # Create the dataframe
    df = pd.DataFrame(data=data, columns=headers)

    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False, mode='w')

    print("Saved: "  + filepath)


def save_values_to_csv(data_dir, filename, values):

    filepath = os.path.join(data_dir, filename)
    with open(filepath, 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write multiple rows
        writer.writerows(values)

        print(filepath + " saved")


def load_solutions_all (data_dir, filename):
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath)
    return df




def cluster_DBSCAN (df):
    from sklearn.cluster import DBSCAN, SpectralClustering
    from sklearn.mixture import GaussianMixture
    from numpy import unique, where

    #init
    # clustering = DBSCAN(eps = 5, min_samples = 2)
    #clustering = SpectralClustering(n_clusters=6)
    clustering  = GaussianMixture(n_components=3)

    # fit model and predict clusters
    yhat = clustering.fit_predict(df)

    # retrieve unique clusters
    clusters = unique(yhat)

    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        dfSample = df.iloc[row_ix]
        sns.scatterplot(data=dfSample, x=df.columns[0], y=df.columns[1])

    # show the plot
    plt.show()


def eval_unknown_property (list_a, list_b, n, list_property_eval_func, list_unknown_property, list_property_unknown_eval_distr):
    # Sample from the optimised dist. of parameters
    list_samples = [np.random.beta(list_a[i], list_b[i], size=n) for i in range(len(list_a))]

    #find the number of target property distributions
    dist_len = len(list_unknown_property)

    fig, axes = plt.subplots(1, dist_len, figsize=(dist_len * 5, 4))


    for i in range(0, dist_len):
        # Sample from the PK
        samples_tar = list_unknown_property[i].sample(n)

        # Calculate statistics (moments)
        sample_mean = np.mean(samples_tar)
        sample_variance = np.var(samples_tar)

        #fit distribution
        if list_property_unknown_eval_distr[i] is Property_Dist.Beta:
            fit_a, fit_b = beta_from_moments(sample_mean, sample_variance)
        elif list_property_unknown_eval_distr[i] is Property_Dist.Gamma:
            fit_a, fit_b = gamma_from_moments(sample_mean, sample_variance)
        print("Target:", fit_a, fit_b)

        # Calulate the sample of y via the sample of x
        sampled_Y = list_property_eval_func[i](list_samples)

        # Calculate statistics (moments)
        sample_mean = np.mean(sampled_Y)
        sample_variance = np.var(sampled_Y)

        if list_property_unknown_eval_distr[i] is Property_Dist.Beta:
            fit_a, fit_b = beta_from_moments(sample_mean, sample_variance)
            sampled_dist = tfd.Beta(tf.constant(fit_a, dtype=tf.float32), tf.constant(fit_b, dtype=tf.float32))
        elif list_property_unknown_eval_distr[i] is Property_Dist.Gamma:
            fit_a, fit_b = gamma_from_moments(sample_mean, sample_variance)
            sampled_dist = tfd.Gamma(tf.constant(fit_a, dtype=tf.float32), tf.constant(fit_b, dtype=tf.float32))

        print("Sampled:", fit_a, fit_b)
        kl = tfd.kl_divergence(list_unknown_property[i], sampled_dist)
        print("KL:", kl.numpy())

        # Plot a histogram of the samples from the true and estimated distributions
        if (dist_len == 1):
            axis = axes
        else:
            axis = axes[i]
        sns.histplot(ax=axis, data=samples_tar, kde=True, label="PK")
        sns.histplot(ax=axis, data=sampled_Y, kde=True, label="Optimised")

    plt.show()

def save_stats_all(stats):
    return 0

def save_stats (stats, data_list):
    for data_item in data_list:
        print (stats.select(data_item))
    return 0


def find_best_from_Pareto (data_dir, filename, property_eval_func, property_type, n=10**5):
    #load Pareto front solutions
    df = load_solutions_all(data_dir, filename)

    #find column names of variables
    variables = [c for c in df.columns.tolist() if "v" in c]


    for i in df.index:
        if property_type is Property_Dist.Beta:
            variable_values = df.loc [i, variables]

            list_a = [variable_values[i] for i in range(len(variables)) if i % 2 == 0]
            list_b = [variable_values[i] for i in range(len(variables)) if i % 2 == 1]

            list_samples = [np.random.beta(list_a[i], list_b[i], size=n) for i in range(len(list_a))]

            # Sample for the property of interest given the generated values
            sampled_Y = property_eval_func(list_samples)

            list_func = [variable_values[i]/(variable_values[i]+variable_values[i+1])
                            for i in range(int(len(variables)/2))]
            sampled_func = property_eval_func(list_func)

            print("%.3f \t %.3f \t %.3f" %
                  (np.median(sampled_Y), np.mean(sampled_Y), sampled_func))



def find_reference_front_and_point (data_dir, pareto_file_substring):
    """
    Find and save the reference fron and the reference point for the statistical analysis
    :param data_dir: data where the Pareto front and set files are saved
    :param pareto_file_substring: the prefix of the Pareto front files that will be used for the calculation
    :return:
    """
    from deap import base
    from deap import creator
    from deap import tools

    REFERENCE_FRONT_FILE = 'Reference_Front.csv'
    REFERENCE_POINT_FILE = 'Reference_Point.csv'
    PARETOSOLUTIONS_FILE = "ParetoFrontSet-All.csv"

    pf_files = get_files_using_substring(data_dir, pareto_file_substring)

    pf_files = [f for f in pf_files if PARETOSOLUTIONS_FILE not in f]

    solutions = None

    for pf_file in pf_files:
        if solutions is None:
            solutions = load_solutions_all("", pf_file)
        else:
            pf        = load_solutions_all("", pf_file)
            solutions = pd.concat([solutions, pf], axis=0)

    POPULATION  = solutions.shape[0]
    NVARS       = np.sum([i.startswith('v') for i in solutions.columns])
    NOBJ        = solutions.shape[1] - NVARS
    BOUND_LOW   = 0
    BOUND_UP    = 1

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
    creator.create("Individual", list, fitness=creator.FitnessMin)
    # define the classes for the individual and population
    toolbox = base.Toolbox()
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NVARS)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("selectFront", tools.sortNondominated)

    population = toolbox.population(n=POPULATION)

    for i in range(POPULATION):
        pf_values = solutions.iloc[i][NVARS:]
        population[i].fitness.setValues(pf_values)

        for j in range(NVARS):
            population[i][j] = solutions.iloc[i][j]


    reference_front = toolbox.selectFront(population, len(population), first_front_only=True)

    reference_front = reference_front[0]
    save_solutions_all_csv(data_dir, REFERENCE_FRONT_FILE, reference_front)
    save_solutions_all_csv(data_dir, PARETOSOLUTIONS_FILE, population)

    #Find reference point for HV calculation
    reference_point_HV = np.max([i.fitness.values for i in population], axis=0)
    reference_point_HV = np.vstack([solutions.columns[NVARS:], reference_point_HV])
    save_values_to_csv(data_dir, REFERENCE_POINT_FILE, reference_point_HV)

    from EPIK_visualisation import show_scatterPlot_Front
    show_scatterPlot_Front(reference_front, population)
    return REFERENCE_FRONT_FILE, REFERENCE_POINT_FILE


def get_files_using_substring(dir_path, substring):
    import os

    # list to store files
    res = []

    # Iterate directory
    for file_path in os.listdir(dir_path):
        if substring in file_path:
            #print(file_path)
            res.append(os.path.join(dir_path, file_path))

    #print(res)
    return res


if __name__ == '__main__':
    data_dir = "data/FPR_R1R2_p2p3/ParetoFrontSet"
    filenames = [
                    "ParetoFrontSetNSGAII-11.csv",
                    "ParetoFrontSetSPEA2-12.csv",
                    # "ParetoFrontSetNSGAII-20.csv"
    ]

    dfData = []
    #dfCluster = df.iloc[:,-2:]
    #cluster_DBSCAN(dfCluster)

    #print(df)


    #find_best_from_Pareto(filename, PRESTO_reqs.property_R3_unknown_p3, Property_Dist.Beta)

    # pf_files = ["data/Solutions.csv", "data/Solutions1.csv", "data/Solutions2.csv"]
    # find_reference_front(pf_files)
    # pf_files = get_files_using_substring("data", "ParetoFrontSet")
    # print(pf_files)

    # data_dir = "data"
    # substring = "ParetoFrontSet"
    # find_reference_front_and_point(data_dir, substring)
