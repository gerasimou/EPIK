import numpy as np
import seaborn as sns
import tensorflow_probability
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random

from EPIK_Utils import Property_Dist, beta_from_moments, gamma_from_moments

tfd = tensorflow_probability.distributions


random.seed(64)



class EPIK:
    def __init__(self, samples_n, list_property_eval_func, list_property_eval_distr, list_property_prior_knowledge):
        self.n                              = samples_n
        self.list_property_eval_func        = list_property_eval_func
        self.list_property_prior_knowledge  = list_property_prior_knowledge
        self.list_property_eval_distr       = list_property_eval_distr
        self.optimisation_iter              = 0


    #Return the distribution of y, given the distribution of x
    def y_dist_from_x(self, list_a, list_b, n, property_eval_func, prop_distr = Property_Dist.Beta):
      # Inputs are the two Beta distribution parameters of x
      # list_a: the alpha parameters of Beta distribution using the traditional parameterisation for the parameters of interest
      # list_b: the beta parameter of Beta distribution using the traditional parameterisation for the parameters of interest
      # n: size of samples
      # property_eval_func: the property function to be evaluated (sampled)

      # Generate n samples from the beta distribution for the three parameters (as many as the size of the list)
      list_samples = [ np.random.beta(list_a[i], list_b[i], size=n) for i in range(len(list_a)) ]

      # Sample for the property of interest given the generated values
      sampled_Y    = property_eval_func (list_samples)

      # Calculate statistics (moments)
      sample_mean     = np.mean(sampled_Y)
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
        # return stats.gamma(scale=1/fit_b, a=fit_a)


    # Loss function
    def loss_func(self, list_a, list_b):
        # list_a: the alpha parameters of Beta distribution using the traditional parameterisation for the parameters of interest
        # list_b: the beta parameter of Beta distribution using the traditional parameterisation for the parameters of interest
        # list_property_eval_func: list of the functions (properties) of interest

        # Calculate the KL divergence between the two distributions
        kl_divergence_list = []
        for i in range(len(self.list_property_eval_func)):
            sampled_distr = self.y_dist_from_x(list_a, list_b, self.n, self.list_property_eval_func[i],
                                               self.list_property_eval_distr[i])
            # KL divergenece using tfd method
            kl_divergence = tfd.kl_divergence(self.list_property_prior_knowledge[i], sampled_distr)
            # KL divergence implementation
            # kl_divergence = self.kl_divergence_func2(self.list_property_prior_knowledge[i], sampled_distr,
            #                                          self.list_property_eval_distr[i])
            kl_divergence_list.append(kl_divergence)

        #     print(kl_divergence_list)
        return np.sum(kl_divergence_list)


    def run_optimisation(self, parameters, bounds, method_name):
        # Use the different algorithms to minimise the objective function and log the progress
        result = minimize(self.objective_eval, parameters, method=method_name, bounds=bounds, callback=self.callback)

        # Print the optimal values for the parameters and the minimum value of the objective function
        print("Optimal parameters:", result.x)
        print("Minimum objective value:", result.fun)

        print(result)

        return result


    # Define the objective function to minimise
    def objective_eval(self, x):
        list_a = [x[i] for i in range(len(x)) if i % 2 == 0]
        list_b = [x[i] for i in range(len(x)) if i % 2 == 1]

        return self.loss_func(list_a, list_b)

        # For printing the progression information of the optimisation


    def callback(self, x):
        print("Iteration", self.optimisation_iter, ":", "x =", x, "obj =", self.objective_eval(x))
        self.optimisation_iter += 1

        # Function to show the compared plot for the distribution of properties (with prior knowledge vs estimated)


    def showPlot(self, list_a, list_b):
        plots_num = len(self.list_property_eval_func)

        fig, axes = plt.subplots(1, plots_num, figsize=(plots_num * 5, 4))

        # Sample from the optimised dist. of parameters
        list_samples = [np.random.beta(list_a[i], list_b[i], size=n) for i in range(len(list_a))]

        for i in range(0, plots_num):
            # Sample from the PK
            samples_tar = self.list_property_prior_knowledge[i].sample(n)

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


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     from PRESTO_reqs import *
#
#     # Number of samples
#     n = 50000
#
#     # list keeping the formulae of properties for which we have prior knowledge
#     list_property_eval_func = [property_R1_unknown_p3, property_R2_unknown_p3]
#
#     # list keeping the type of properties for which we have prior knowledge
#     # Property_Dist.Beta -> [0,1], Property_Dist.Gamma -> [0,\infty)
#     list_property_eval_distr = [Property_Dist.Beta, Property_Dist.Gamma]
#
#     # list keeping the prior knowledge for the properties of interest
#     list_property_prior_knowledge = [dist_PK_R1, dist_PK_R2_conflictC]
#
#     # the initial guess for the parameters
#     x0 = np.array([2.0, 2.0])  # , 2.0, 2.0])
#
#     # the bounds for the parameters (optional)
#     bounds = [(0.00001, 1000), (0.00001, 1000)]  # ,(0.00001, 1000),(0.00001, 1000)]
#
#     # initialise the EPIK instance
#     epik1 = EPIK(n, list_property_eval_func, list_property_eval_distr, list_property_prior_knowledge)
#
#     # execute the optimisation
#     result1 = epik1.run_optimisation(x0, bounds, "powell")
#
#     # Prepare lists
#     list_a = [result1.x[i] for i in range(len(result1.x)) if i % 2 == 0]
#     list_b = [result1.x[i] for i in range(len(result1.x)) if i % 2 == 1]
#
#     # Print plot
#     epik1.showPlot(list_a, list_b)




if __name__ == '__main__':
    from ReqsFX import *

    # Number of samples
    n = 100000

    # list keeping the formulae of properties for which we have prior knowledge
    list_property_eval_func = [property_R1_unknown_p51]#, property_R2_unknown_p5]

    # list keeping the type of properties for which we have prior knowledge
    # Property_Dist.Beta -> [0,1], Property_Dist.Gamma -> [0,\infty)
    list_property_eval_distr = [Property_Dist.Beta]#, Property_Dist.Gamma]

    # list keeping the prior knowledge for the properties of interest
    list_property_prior_knowledge = [dist_PK_R1]#, dist_PK_R2_conflictC]

    # the initial guess for the parameters
    x0 = np.array([2.0, 2.0])#, [2.0, 2.0])

    # the bounds for the parameters (optional)
    bounds = [(0.00001, 1000), (0.00001, 1000)]  # ,(0.00001, 1000),(0.00001, 1000)]

    # initialise the EPIK instance
    epik1 = EPIK(n, list_property_eval_func, list_property_eval_distr, list_property_prior_knowledge)

    # execute the optimisation
    result1 = epik1.run_optimisation(x0, bounds, "powell")

    # Prepare lists
    list_a = [result1.x[i] for i in range(len(result1.x)) if i % 2 == 0]
    list_b = [result1.x[i] for i in range(len(result1.x)) if i % 2 == 1]

    # Print plot
    epik1.showPlot(list_a, list_b)



