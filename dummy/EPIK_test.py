import EPIK_old
from EPIK_Utils import *
# from PRESTO_reqs import  *


# def example_function(a, b):
#     return a + b
#
# func = getattr(EPIK_Utils, 'uniform')
# print(func(0, 1, 3))  # 👉️ 'abcd'


#Distance
# calculate the kl divergence
def kl_divergence(p, q):
    from math import log2, log
    return sum(p[i] * log(p[i] / q[i]) for i in range(len(p)))

def kl_divergence2(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))



def kl_continuous2(P, Q, lower=-np.inf, upper=np.inf):
    from scipy.integrate import quad
    # Define the integrand
    def integrand(x):
        return P.pdf(x) * np.log(P.pdf(x) / Q.pdf(x))

    # Calculate the integral
    return quad(integrand, lower, upper)[0]

# def kl_continuous (P, Q):
#     from scipy.special import digamma
#     d1_concentration1 = P.kwds['b']
#     d1_concentration0 = P.kwds['a']
#     d2_concentration1 = Q.kwds['b']
#     d2_concentration0 = Q.kwds['a']
#     d1_total_concentration = d1_concentration1 + d1_concentration0
#     d2_total_concentration = d2_concentration1 + d2_concentration0
#
#     d1_log_normalization = d1._log_normalization(  # pylint: disable=protected-access
#         d1_concentration1, d1_concentration0)
#     d2_log_normalization = d2._log_normalization(  # pylint: disable=protected-access
#         d2_concentration1, d2_concentration0)
#     return ((d2_log_normalization - d1_log_normalization) -
#             (digamma(d1_concentration1) *
#              (d2_concentration1 - d1_concentration1)) -
#             (tf.math.digamma(d1_concentration0) *
#              (d2_concentration0 - d1_concentration0)) +
#             (digamma(d1_total_concentration) *
#              (d2_total_concentration - d1_total_concentration)))


if __name__ == '__main__':
    #Number of samples
    n = 1000
    #
    # #list of unknown properties
    # list_property_unknown         = [dist_PK_R2]
    #
    # list_property_unknown_eval_func = [property_R2_unknown_p3]
    #
    # list_property_unknown_eval_distr = [Property_Dist.Gamma]
    #
    #
    # # results
    # list_a = [5.1386]
    # list_b = [13.9174]
    #
    # eval_unknown_property(list_a, list_b, n, list_property_unknown_eval_func, list_property_unknown, list_property_unknown_eval_distr)



    # from FX_reqs import *
    # from scipy import stats
    #
    # # property_R2_unknown_p51
    # #dist_PK_R1
    #
    # d1 = dist_PK_R1b
    #
    # b1_a = 475
    # b1_b = 25
    # d2 = stats.beta(a=b1_a, b=b1_b)
    #
    #
    # # d1_sample = d1.rvs(size=n)
    # # d2_sample = d2.rvs(size=n)
    # klDiv = kl_continuous(d2, d2)
    # print(klDiv)
    #
    # from tensorflow_probability import distributions as tfd
    #
    # d1 = dist_PK_R1
    # d2 = tfd.Beta(tf.constant(b1_a, dtype=tf.float32), tf.constant(b1_b, dtype=tf.float32))
    #
    # klDiv = tfd.kl_divergence(d1, d2)
    # print(klDiv.numpy())


    import ReqsFX
    from EPIK_Utils import Property_Dist, eval_property, load_solutions_all
    from EPIK_visualisation import show_distPlot2
    problem_name =  "FX_R1R2_p5_1000b"
    n = 100000
    list_property_eval_func = [FX_reqs.property_R1_unknown_p51, FX_reqs.property_R2_unknown_p51]
    list_property_prior_knowledge = [FX_reqs.dist_PK_R1b, FX_reqs.dist_PK_R2b]
    list_property_eval_distr = [Property_Dist.Beta, Property_Dist.Gamma]

    data_dir = "data" +"/"+ problem_name + "/ParetoFrontSet"
    dataFilename = "ParetoFrontSet-All.csv"
    outFilename = data_dir +"/"+ problem_name +"_"
    labels = ["R1", "R2"]


    #load Pareto front and get the first entry
    df = load_solutions_all(data_dir, dataFilename)
    results = df.iloc[9][:2].to_numpy()
    KLvalue = df.iloc[9][-1]

    # Prepare lists
    list_a = [results[i] for i in range(len(results)) if i % 2 == 0]
    list_b = [results[i] for i in range(len(results)) if i % 2 == 1]

    title = problem_name +"  ("+ str(KLvalue) +")"
    show_distPlot2  (list_a, list_b, list_property_eval_func, n, list_property_prior_knowledge, outFilename, labels, title)



    res = eval_property(list_property_eval_distr[0], list_property_eval_func[0], results, n)
    print(res, "\t", np.mean(res))



