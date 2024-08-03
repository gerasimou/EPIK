from tensorflow_probability import distributions as tfd
import tensorflow as tf
from scipy import stats
from scipy.spatial import distance
from math import log, log2, sqrt

import numpy as np


#Knowledge
b1_a = 160
b1_b = 40
n = 10000

#Distributions
b1   = tfd.Beta(tf.constant(b1_a, dtype=tf.float32), tf.constant(b1_b, dtype=tf.float32))
b1b  = stats.beta(a=b1_a, b=b1_b)
b1bb = tfd.Beta(tf.constant(b1_a, dtype=tf.float32), tf.constant(b1_b, dtype=tf.float32))

#Sampling
b1Samples   = b1.sample(n).numpy()
b1bSamples  = b1b.rvs(n)
b1bbSamples = b1bb.sample(n).numpy()


#Distance
# calculate the kl divergence
def kl_divergence(p, q):
    return sum(p[i] * log2(p[i] / q[i]) for i in range(len(p)))


# calculate the js divergence
def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)



def js_dist(P, Q):
    """Compute the Jensen-Shannon distance between two probability distributions.

    Input
    -----
    P, Q : array-like
        Probability distributions of equal length that sum to 1
    """

    def _kldiv(A, B):
        # Calculate Kullback-Leibler divergence

        return np.sum([v for v in A * np.log2(A / B) if not np.isnan(v)])

    P = np.array(P)
    Q = np.array(Q)

    M = 0.5 * (P + Q)

    # Get the JS DIVERGENCE
    result = 0.5 * (_kldiv(P, M) + _kldiv(Q, M))
    # Take sqrt to get the JS DISTANCE
    return np.sqrt(result)


def hellinger_distance (p, q):
    return sqrt(sum([(sqrt(p_i) - sqrt(q_i)) ** 2 for p_i, q_i in zip(p, q)]) / 2)


#Check with itself
print("JS distance Same distribution (Function)")
print(distance.jensenshannon(b1Samples,    b1Samples))
print(distance.jensenshannon(b1bSamples,   b1bSamples))
print(distance.jensenshannon(b1bbSamples,  b1bbSamples))

print("JS distance Same distribution  (MANUAL)")
print(np.sqrt(js_divergence(b1Samples,    b1Samples)),   js_dist(b1Samples,    b1Samples))
print(np.sqrt(js_divergence(b1bSamples,   b1bSamples)),  js_dist(b1bSamples,   b1bSamples))
print(np.sqrt(js_divergence(b1bbSamples,  b1bbSamples)), js_dist(b1bbSamples,  b1bbSamples))


print("\nJS Comparison (Function)")
print(distance.jensenshannon(b1Samples,  b1bSamples))
print(distance.jensenshannon(b1Samples,  b1bbSamples))
print(distance.jensenshannon(b1bSamples,  b1bbSamples))

print("\nJS Comparison (MANUAL)")
print(np.sqrt(js_divergence(b1Samples,  b1bSamples)),   js_dist(b1Samples,    b1bSamples))
print(np.sqrt(js_divergence(b1Samples,  b1bbSamples)),  js_dist(b1Samples,    b1bbSamples))
print(np.sqrt(js_divergence(b1bSamples,  b1bbSamples)), js_dist(b1bSamples,   b1bbSamples))


print("\nWhole Distribution Comparison")
r = tfd.kl_divergence(b1, b1bb).numpy()
print(r)


print("\nHellinger distance")
print(hellinger_distance(b1bbSamples,  b1bbSamples))
print(hellinger_distance(b1Samples,  b1bSamples))
print(hellinger_distance(b1Samples,  b1bbSamples))
print(hellinger_distance(b1bSamples,  b1bbSamples))


print("\nKL Divergence")
print(kl_divergence(b1bbSamples,  b1bbSamples))
print(kl_divergence(b1Samples,  b1bSamples))
print(kl_divergence(b1Samples,  b1bbSamples))
print(kl_divergence(b1bSamples,  b1bbSamples))



def hellinger_distance_beta (a1, b1, a2, b2):
    from scipy.special import beta as beta_func
    dist = 1 - beta_func( (a1+a2)/2, (b1+b2)/2 ) / sqrt( beta_func(a1, b1) * beta_func(a2, b2) )
    return sqrt(dist)

def hellinger_distance_gamma (a1, b1, a2, b2):
    from scipy.special import gamma as gamma_func
    dist = 1 - gamma_func((a1+a2)/2) * \
           ( ((b1+b2)/2)**(-(a1+a2)/2) ) * \
           sqrt( ((b1**a1)*(b2**a2))/(gamma_func(a1)*gamma_func(a2)) )
    return sqrt(dist)



# print("\nH distance")
# print(hellinger_distance_beta(b1_a, b1_b, 160, 45))
# print(hellinger_distance_beta(160, 50, b1_a, b1_b))
#
# bc = tfd.Beta(tf.constant(b1_a, dtype=tf.float32), tf.constant(45, dtype=tf.float32))
# bd = tfd.Beta(tf.constant(160, dtype=tf.float32), tf.constant(50, dtype=tf.float32))
#
# print(tfd.kl_divergence(b1, bc).numpy(), tfd.kl_divergence(b1, bd).numpy())
#
#
# g1 = tfd.Gamma(tf.constant(950.0, dtype=tf.float32), tf.constant(200.0, dtype=tf.float32))
# g2 = tfd.Gamma(tf.constant(950.0, dtype=tf.float32), tf.constant(200.0, dtype=tf.float32))
#
#
# print(tfd.kl_divergence(g1, g2).numpy())
# print (hellinger_distance_gamma(950, 200, 950, 201))
#
# print("DONE")


# dist_PK_R1 = tfd.Beta(tf.constant(160.0, dtype=tf.float64), tf.constant(40.0, dtype=tf.float64))


from ReqsPRESTO import dist_PK_R1
a = 56.7676
b = 14.3673
CMAES1 = tfd.Beta(tf.constant(a, dtype=tf.float64), tf.constant(b, dtype=tf.float64))

print("CMAES1: %.3f" % (tfd.kl_divergence(dist_PK_R1, CMAES1).numpy()) )





#print(distance.jensenshannon([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]))


if __name__ == '__main__':
    print("Test")
