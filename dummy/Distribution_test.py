

from EPIK_Utils import kl_divergence_beta
import tensorflow_probability
import tensorflow as tf


tfd = tensorflow_probability.distributions


a1 = 3
b1 = 0.5
a2 = 0.5
b2 = 3

d1 = tfd.Beta(tf.constant(a1, dtype=tf.float32), tf.constant(b1, dtype=tf.float32))
d2 = tfd.Beta(tf.constant(a2, dtype=tf.float32), tf.constant(b2, dtype=tf.float32))

kl_d1_d2 = kl_divergence_beta(d1, d2)
kl_d2_d1 = kl_divergence_beta(d2, d1)

print(kl_d1_d2)
print(kl_d2_d1)