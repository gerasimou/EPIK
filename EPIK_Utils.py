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
from scipy import stats



class Property_Dist (Enum):
    Beta = 0
    Gamma = 1


class MODEL_TYPE(Enum):
    DTMC = 'DTMC'
    CTMC = 'CTMC'


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


def calculate_KL_divergence (list_a, list_b, n, list_property_eval_func, list_property_prior_knowledge, list_property_eval_distr, model_type):
        # Inputs are the two Beta distribution parameters of x
        # list_a: the alpha parameters of Beta distribution using the traditional parameterisation for the parameters of interest
        # list_b: the beta parameter of Beta distribution using the traditional parameterisation for the parameters of interest
        # n: size of samples
        # property_eval_func: the property function to be evaluated (sampled)

        # Generate n samples from the beta distribution for the three parameters (as many as the size of the list)
        #TODO: A change here is needed to support CTMCs - DONE
        if model_type == MODEL_TYPE.DTMC.value:
            list_samples = [np.random.beta(list_a[i], list_b[i], size=n) for i in range(len(list_a))]
        elif model_type == MODEL_TYPE.CTMC.value:
            list_samples = [np.random.gamma(shape=list_a[i], scale=list_b[i], size=n) for i in range(len(list_a))]
        else:
            raise TypeError("Model type not DTMC or CTMC: " + model_type)


        kl_divergence_list = []

        for i in range(len(list_property_eval_func)):

            # Sample for the property of interest given the generated values
            sampled_Y = list_property_eval_func[i](list_samples)

            # Calculate statistics (moments)
            sample_mean = np.mean(sampled_Y)
            sample_variance = np.var(sampled_Y)

            # Calculate distributions parameters from moments,
            # depending on its type and return the distributions
            prop_distr = list_property_eval_distr[i]
            if prop_distr is Property_Dist.Beta:
                fit_a, fit_b = beta_from_moments(sample_mean, sample_variance)
                sampled_distr =  tfd.Beta(tf.constant(fit_a, dtype=tf.float32), tf.constant(fit_b, dtype=tf.float32))
                # sampled_distr = stats.beta(a=fit_a, b=fit_b)
                kl_divergence = kl_divergence_beta(list_property_prior_knowledge[i], sampled_distr)
            elif prop_distr is Property_Dist.Gamma:
                fit_a, fit_b = gamma_from_moments(sample_mean, sample_variance)
                sampled_distr =  tfd.Gamma(tf.constant(fit_a, dtype=tf.float32), tf.constant(fit_b, dtype=tf.float32))
                # sampled_distr = stats.gamma(scale=1 / fit_b, a=fit_a)
                kl_divergence = kl_divergence_gamma(list_property_prior_knowledge[i], sampled_distr)


            print(kl_divergence.numpy())
            kl_divergence_list.append(kl_divergence)

        return np.sum(kl_divergence_list)


def kl_divergence_beta(d1, d2):
    d1_concentration1 = tf.convert_to_tensor(d1.concentration1)
    d1_concentration0 = tf.convert_to_tensor(d1.concentration0)
    d2_concentration1 = tf.convert_to_tensor(d2.concentration1)
    d2_concentration0 = tf.convert_to_tensor(d2.concentration0)
    d1_total_concentration = d1_concentration1 + d1_concentration0
    d2_total_concentration = d2_concentration1 + d2_concentration0

    d1_log_normalization = d1._log_normalization(  # pylint: disable=protected-access
        d1_concentration1, d1_concentration0)
    d2_log_normalization = d2._log_normalization(  # pylint: disable=protected-access
        d2_concentration1, d2_concentration0)

    d1_concentration1      = d1_concentration1.numpy()
    d1_concentration0      = d1_concentration0.numpy()
    d2_concentration1      = d2_concentration1.numpy()
    d2_concentration0      = d2_concentration0.numpy()
    d1_total_concentration = d1_total_concentration.numpy()
    d2_total_concentration = d2_total_concentration.numpy()
    d1_log_normalization   = d1_log_normalization.numpy()
    d2_log_normalization   = d2_log_normalization.numpy()

    vv = ((d2_log_normalization - d1_log_normalization) -
          (tf.math.digamma(d1_concentration1) *
           (d2_concentration1 - d1_concentration1)) -
          (tf.math.digamma(d1_concentration0) *
           (d2_concentration0 - d1_concentration0)) +
          (tf.math.digamma(d1_total_concentration) *
           (d2_total_concentration - d1_total_concentration)))

    return vv


def kl_divergence_gamma(g0, g1):
    g0_concentration = tf.convert_to_tensor(g0.concentration)
    g0_log_rate = g0._log_rate_parameter()  # pylint: disable=protected-access
    g1_concentration = tf.convert_to_tensor(g1.concentration)
    g1_log_rate = g1._log_rate_parameter()  # pylint: disable=protected-access

    g0_concentration = g0_concentration.numpy()
    g0_log_rate      = g0_log_rate.numpy()
    g1_concentration = g1_concentration.numpy()
    g1_log_rate      = g1_log_rate.numpy()

    return (((g0_concentration - g1_concentration) *
             tf.math.digamma(g0_concentration)) +
            tf.math.lgamma(g1_concentration) -
            tf.math.lgamma(g0_concentration) +
            g1_concentration * g0_log_rate -
            g1_concentration * g1_log_rate +
            g0_concentration * tf.math.expm1(g1_log_rate - g0_log_rate))


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

    return filepath


def save_values_to_csv(data_dir, filename, values):

    filepath = os.path.join(data_dir, filename)
    with open(filepath, 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write multiple rows
        writer.writerows(values)

        print(filepath + " saved")

    return filepath


def save_solution_all_single (filename, solution):
    """
    Save the solution after a single objective EPIK executionb
    :param data_dir:
    :param filename:
    :param solution:
    :param append: append to the specified @file
    :return:
    """

    # Find the number of properties
    objsNum = len(solution.fitness.values)

    # Find the number of solution items
    solItems = len(solution)

    # Create the column headers
    headers = ["v" + str(i + 1) for i in range(solItems)]
    colHeaders = ["P" + str(i + 1) for i in range(objsNum)]
    headers.extend(colHeaders)
    #print(headers)


    # Create data to save
    data = [solution[i] for i in range(solItems)]
    data.extend(solution.fitness.values)

    #if the file does not exist, create one
    if not os.path.isfile(filename):
        # Create the dataframe
        dataLoL = [data]
        df = pd.DataFrame(data=dataLoL, columns=headers)
        df.to_csv(filename, index=False, mode='w')
    else: #simply write to the file
        with open(filename, "a", newline='') as fp:
            wr = csv.writer(fp)
            wr.writerow(data)


    print("Saved: "  + filename)
    return filename



def save_KL_history(data_dir, filename, KL_storage):
    """
    Save the evolution of KL values across the population and generations
    :param filename:
    :param KL_storage:
    :return:
    """

    with open(filename, 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write multiple rows
        writer.writerows(KL_storage)

        print(filename + " saved")

    return filename


def load_solutions_all (data_dir, filename):
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath)
    return df



def eval_property (property_type, property_func, params, n):
    if property_type is Property_Dist.Beta:
        eval_param_dist = tfd.Beta(tf.constant(params[0], dtype=tf.float32), tf.constant(params[1], dtype=tf.float32))
    elif property_type is Property_Dist.Gamma:
        eval_param_dist = tfd.Gamma(tf.constant(params[0], dtype=tf.float32), tf.constant(params[1], dtype=tf.float32))

    samples_param_dist = eval_param_dist.sample(n)


    samples_dist = property_func([samples_param_dist.numpy()])

    return  samples_dist




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


def save_stats (stats, data_list):
    for data_item in data_list:
        print (stats.select(data_item))
    return 0



def embed_knowledge (data_dir, filename, property_eval_func, n=10 ** 6):
    """
    Embed knowledge to establish the values of the unknown (elusive) property
    :param data_dir:
    :param filename:
    :param property_eval_func:
    :param property_type:
    :param n:
    :return:
    """

    #load Pareto front solutions
    df = load_solutions_all(data_dir, filename)
    df = df.loc[:19, :]

    #find column names of variables
    variables = [c for c in df.columns.tolist() if "v" in c]

    df_elusive_props = pd.DataFrame(columns=['EP1_median', 'EP1_mean', 'EP1_std'])

    import seaborn as sns
    import matplotlib.pyplot as plt
    cols = 5
    rows = int(np.ceil((df.shape[0]+1)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10))

    for i in df.index:
        variable_values = df.loc [i, variables]

        list_a = [variable_values[i] for i in range(len(variables)) if i % 2 == 0]
        list_b = [variable_values[i] for i in range(len(variables)) if i % 2 == 1]

        #sample
        list_samples = [np.random.beta(list_a[i], list_b[i], size=n) for i in range(len(list_a))]

        # Sample for the property of interest given the generated values
        sampled_Y = property_eval_func(list_samples)

        # #calculate property result using the mean of the beta function
        # list_func = [variable_values[i]/(variable_values[i]+variable_values[i+1])
        #                 for i in range(int(len(variables)/2))]
        # sampled_func = property_eval_func(list_func)

        print("%i \t %.3f \t %.3f \t %.3f" % (i, np.median(sampled_Y), np.std(sampled_Y), np.mean(sampled_Y)))

        results_i = [np.median(sampled_Y), np.mean(sampled_Y), np.std(sampled_Y)]
        df_elusive_props.loc[i] = results_i

        j = int(i/5)
        k = i%5
        ax = sns.histplot(ax=axes[j, k], data=sampled_Y, kde=True, label="Optimised")
        # ax.legend(loc='best', frameon=False)
    plt.show()
    print(df_elusive_props)



def embed_knowledge_b (data_dir, filename, property_eval_func, n=10 ** 6):
    #load Pareto front solutions
    df = load_solutions_all(data_dir, filename)
    df = df.loc[:19, :]

    #find column names of variables
    variables = [c for c in df.columns.tolist() if "v" in c]

    df_elusive_props_moments = pd.DataFrame(columns=['EP1_median', 'EP1_mean', 'EP1_std'])
    df_elusive_props_results = pd.DataFrame(columns=[])

    import seaborn as sns
    import matplotlib.pyplot as plt
    cols = 4
    rows = 1#int(np.ceil((df.shape[0]+1)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10))

    for i in df.index:
        variable_values = df.loc [i, variables]

        list_a = [variable_values[i] for i in range(len(variables)) if i % 2 == 0]
        list_b = [variable_values[i] for i in range(len(variables)) if i % 2 == 1]

        #sample
        list_samples = [np.random.beta(list_a[i], list_b[i], size=n) for i in range(len(list_a))]

        # Sample for the property of interest given the generated values
        sampled_Y = property_eval_func(list_samples)

        moments_i = [np.median(sampled_Y), np.mean(sampled_Y), np.std(sampled_Y)]

        #append to dfs
        df_elusive_props_moments.loc[i] = moments_i
        df_elusive_props_results["ID"+str(i)] = sampled_Y
        print(i)
        j = int(i/cols)
        k = i%cols

        if rows > 1:
            ax = sns.displot(ax=axes[j, k], data=sampled_Y, kind='kde', label="Optimised")
        else:
            axes[k] = sns.displot(data=sampled_Y, kind='kde', label="Optimised", fill=True)
            axes[k].set(ylim=(0,4.5))

        quants = np.quantile(a=sampled_Y, q=[0.05, 0.5, 0.95]).tolist()
        plt.vlines(x=quants,    # Line on x = 2
                   ymin = [0, 0, 0],  # Bottom of the plot
                   ymax = [4, 4, 4], colors=['r', 'r', 'r']) # Top of the plot
        # ax.legend(loc='best', frameon=False)
        plt.savefig("embedding"+str(i)+".svg", format='svg')
        plt.savefig("embedding"+str(i)+".pdf", format='pdf')

    plt.show()
    print(df_elusive_props_moments)
    return  df_elusive_props_results

def find_reference_front_and_point (data_dir, pareto_file_substring, output_dir):
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
    REFERENCE_FRONT_FILEPATH = save_solutions_all_csv(output_dir, REFERENCE_FRONT_FILE, reference_front)
    PARETOSOLUTIONS_FILEPATH = save_solutions_all_csv(data_dir, PARETOSOLUTIONS_FILE, population)

    #Find reference point for HV calculation
    reference_point_HV = np.max([i.fitness.values for i in population], axis=0)
    reference_point_HV = np.vstack([solutions.columns[NVARS:], reference_point_HV])
    REFERENCE_POINT_FILEPATH = save_values_to_csv(output_dir, REFERENCE_POINT_FILE, reference_point_HV)

    from EPIK_visualisation import show_scatterPlot_Front
    show_scatterPlot_Front(reference_front, population)
    return REFERENCE_FRONT_FILEPATH, REFERENCE_POINT_FILEPATH


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
    data_dir = "data/FPR_R1R2_p3/ParetoFrontSet"
    filenames = [
                    "ParetoFrontSetNSGAII-11b.csv",
                    "ParetoFrontSetSPEA2-12.csv",
                    # "ParetoFrontSetNSGAII-20.csv"
    ]

    dfData = []
    #dfCluster = df.iloc[:,-2:]
    #cluster_DBSCAN(dfCluster)

    #print(df)


    import ReqsPRESTO
    embed_knowledge_b(data_dir, filenames[0], ReqsPRESTO.property_R3_unknown_p3)

    # pf_files = ["data/Solutions.csv", "data/Solutions1.csv", "data/Solutions2.csv"]
    # find_reference_front(pf_files)
    # pf_files = get_files_using_substring("data", "ParetoFrontSet")
    # print(pf_files)

    # data_dir = "data"
    # substring = "ParetoFrontSet"
    # find_reference_front_and_point(data_dir, substring)
