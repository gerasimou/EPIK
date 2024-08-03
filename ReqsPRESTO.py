from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


##Functions for properties R1 - R3
#R1: Success probability - [0,1]
def property_R1(p1, p2, p3):
    return (p1 - p1*p2) / (1 - p1*p2*p3)


#p1=0.9, p2=0.3
def property_R1_unknown_p3(list_samples):
        return (0.63)/(1 - 0.27*list_samples[0])


#p1=0.9
def property_R1_unknown_p2p3(list_samples):
            return (0.9 - 0.9*list_samples[0]) / (1 - 0.9*list_samples[0]*list_samples[1])

#all unknown
def property_R1_unknown_p1p2p3(list_samples):
    return (list_samples[0] - list_samples[0] * list_samples[1]) / \
        (1 - list_samples[0] * list_samples[1] * list_samples[2])


#R2 Time completion - [0, infty)
def property_R2(p1, p2, p3):
    return (3*p1*p2 + 1 + 2*p1) / (1 - p1*p2*p3)

#p1=0.9, p2=0.3
def property_R2_unknown_p3(list_samples):
    return (3.61)/(1-0.27*list_samples[0])

#p1=0.9
def property_R2_unknown_p2p3(list_samples):
    return (2.7*list_samples[0] + 2.8) / (1 - 0.9*list_samples[0]*list_samples[1])

#all unknown
def property_R2_unknown_p1p2p3(list_samples):
    return (3*list_samples[0]*list_samples[1] + 1 + 2*list_samples[0]) / \
        (1 - list_samples[0]*list_samples[1]*list_samples[2])


#R3 Energy overheads - [0, infty)
def property_R3(p1, p2, p3):
    return (p1*p2 + 1 + p1) / (1 - p1*p2*p3)

#p1=0.9, p2=0.3
def property_R3_unknown_p3(list_samples):
    return (2.17) / (1-0.27*list_samples[0])

#p1=0.9
def property_R3_unknown_p2p3(list_samples):
    return (0.9*list_samples[0] + 1.9) / (1 - 0.9*list_samples[0]*list_samples[1])

#all unknown
def property_R3_unknown_p1p2p3(list_samples):
    return (list_samples[0]*list_samples[1] + 1 + list_samples[0]) / \
        (1 - list_samples[0]*list_samples[1]*list_samples[2])



# Prior knowledge of R1, as a Beta distribution with given parameters
#using Tensoflow distribution
dist_PK_R1 = tfd.Beta(tf.constant(160.0, dtype=tf.float64), tf.constant(40.0, dtype=tf.float64))
#using Scipy
dist_PK_R1b = stats.beta(a=160, b=40)


# Prior knowledge of R2, as a Gamma distribution with given parameters
#using Tensoflow distribution
dist_PK_R2 = tfd.Gamma(tf.constant(950.0, dtype=tf.float32), tf.constant(200.0, dtype=tf.float32))
#using Scipy
dist_PK_R2b = stats.gamma(scale=1/200, a=950)

#conflicting P2
dist_PK_R2_conflict  = tfd.Gamma(tf.constant(251.93, dtype=tf.float32), tf.constant(57.85, dtype=tf.float32))
dist_PK_R2b_conflict = stats.gamma(scale=1/57.85, a=251.93)

dist_PK_R2_conflictb  = tfd.Gamma(tf.constant(326.1075, dtype=tf.float32), tf.constant(81.4742, dtype=tf.float32))
dist_PK_R2b_conflictb = stats.gamma(scale=1/81.4742, a=326.1075)

dist_PK_R2_conflictC = tfd.Gamma(tf.constant(1174.264, dtype=tf.float32), tf.constant(301.373, dtype=tf.float32))
dist_PK_R2b_conflictC = stats.gamma(scale=1/301.373, a=1174.264)

# Prior knowledge of R3, as a Gamma distribution with given parameters
#using Scipy
dist_PK_R3b = stats.gamma(scale=1/100, a=1500)




# def sample




if __name__ == '__main__':
    p1 = 0.9
    p2 = 0.3
    p3 = 0.1

    print (property_R1(p1, p2, p3))
    print (property_R2(p1, p2, p3))
    print (property_R3(p1, p2, p3))