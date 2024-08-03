import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import tensorflow_probability
import tensorflow as tf
tfd = tensorflow_probability.distributions


avgR          = 0.05733023839011232 #8.17338803682312    #4.75
times         = 379682.2739020702   #1.0761375276136629 #200#
concentration = avgR #* times
print(concentration, times)

# distG1 = tfd.Gamma(concentration=concentration, rate=times)
# distG1 = tfd.Beta(concentration1=concentration, concentration0=times)

distG1 = tfd.Gamma(concentration=tf.constant(avgR, dtype=tf.float64), rate=tf.constant(times, dtype=tf.float64))
# sample_Y = distG1.sample(100000)




# import ReqsPRESTO
# vals = distG1.sample(100000)
# sample_Y = ReqsPRESTO.property_R2_unknown_p3([vals])

import ReqsDPM
vals = distG1.sample(100000)
sample_Y = ReqsDPM.property_R1_unknown_evTrans11([vals])

sample_mean = np.mean(sample_Y)
sample_median = np.median(sample_Y)
sample_variance = np.var(sample_Y)
print(sample_mean, sample_median, sample_variance)


# sns.histplot(data=sample_Y, kde=True, label="PK")
# plt.show()


# sample_mean = 1.85
# sample_variance = 1**(-6)
g0 = tfd.Gamma.experimental_from_mean_variance(sample_mean, sample_variance)


print("DONE")