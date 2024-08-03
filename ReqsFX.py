from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


#R1: Probability of successfully completing the workflow execution
#
#p51? [0.9314, 0.9535]
def property_R1_unknown_p51(list_samples):
    probOp51 = list_samples[0]
    res =  (-11111 * (24787109256203850*probOp51+(-1069720330268012377)))/(12464547117895500000000)
    # res = (0.63) / (1 - 0.27 * list_samples[0])
    return res

#p41, p51? [0.8908, 0.9538]
def property_R1_unknown_p41p51(list_samples):
    probOp41 = list_samples[0]
    probOp51 = list_samples[1]
    return (11111 * (1709667987300*probOp41*probOp51+(-57010363501183050)*probOp51+12481854602056854*probOp41+2460267603512128061))/(55281431500000000 * (26614*probOp41+518401))

#p21,p41,p51?
def property_R1_unknown_p21p41p51(list_samples):
    probOp21 = list_samples[0]
    probOp41 = list_samples[1]
    probOp51 = list_samples[2]
    return (33333 * (1452668609700*probOp41*probOp21*probOp51+11398634736612900*probOp51+(-2485367459168412)*probOp41+(-632467319400)*probOp41*probOp51+(-922239304350771)*probOp21+17189818118550*probOp21*probOp51+(-55017306214794)*probOp41*probOp21+(-491869072841555458)))/(500000000 * ((26614*probOp41+518401) * (302621*probOp21+(-66398242))))

#p11,p21,p41,p51?
def property_R1_unknown_p11p21p41p51(list_samples):
    probOp11 = list_samples[0]
    probOp21 = list_samples[1]
    probOp41 = list_samples[2]
    probOp51 = list_samples[3]
    return (99999 * (334648800*probOp11*probOp41*probOp21*probOp51+(-18256194020446245)*probOp51+(-42649967146080)*probOp11*probOp41+110913767280*probOp11*probOp41*probOp51+31974607019880*probOp41*probOp21+(-83151860580)*probOp41*probOp21*probOp51+1441318586622228*probOp41+(-19298324898)*probOp41*probOp51+(-60194760492000)*probOp11*probOp21+111047022000*probOp11*probOp21*probOp51+(-19950550119065200)*probOp11+36804684658200*probOp11*probOp51+14956893113249700*probOp21+(-27592408791450)*probOp21*probOp51+(-128683396800)*probOp11*probOp41*probOp21+7986366610834725570))/(100000000 * ((3802*probOp41+2075505) * (73200*probOp11*probOp21+24260920*probOp11+(-18188370)*probOp21+3971767903)))



#R2: What is the expected response time per workflow execution?
#
#p51? [19.29, 21.965]
def property_R2_unknown_p51(list_samples):
    probOp51 = list_samples[0]
    res = (1198043614049852750*probOp51+8658247774272633577)/(448723696244238000)
    # (39517507925057751*probOp51+1427970364394349802)/(77265616818913200)
    return res

#p41, p51? [18.657, 22.697]
def property_R2_unknown_p41p51(list_samples):
    probOp41 = list_samples[0]
    probOp51 = list_samples[1]
    return (-1 * (105429525883500*probOp41*probOp51+(-3515639082572954750)*probOp51+(-336552054464059254)*probOp41+(-19900691290270099661)))/(1990131534000 * (26614*probOp41+518401))

#p21,p41,p51?
def property_R2_unknown_p21p41p51(list_samples):
    probOp21 = list_samples[0]
    probOp41 = list_samples[1]
    probOp51 = list_samples[2]
    return (-1 * (89581230931500*probOp41*probOp21*probOp51+(-699542112919201629)*probOp21+702915808757795500*probOp51+1060038783977250*probOp21*probOp51+70229355018635612*probOp41+(-14594720629118806)*probOp41*probOp21+(-39002151363000)*probOp41*probOp51+4120046680637860258))/(6000 * ((26614*probOp41+518401) * (302621*probOp21+(-66398242))))

#p11,p21,p41,p51?
def property_R2_unknown_p11p21p41p51(list_samples):
    probOp11 = list_samples[0]
    probOp21 = list_samples[1]
    probOp41 = list_samples[2]
    probOp51 = list_samples[3]
    return (-1 * (92697717600*probOp11*probOp41*probOp21*probOp51+33648050560231586700*probOp21+(-135418253587812000)*probOp11*probOp21+(-8156392908787157200)*probOp11+(-5056965743663609865)*probOp51+(-7643097235231650)*probOp21*probOp51+10194897650321400*probOp11*probOp51+30760025094000*probOp11*probOp21*probOp51+(-635923602544296292)*probOp41+27443223130450680*probOp41*probOp21+(-110446616884800)*probOp11*probOp41*probOp21+(-9112941589518880)*probOp11*probOp41+(-5345635996746)*probOp41*probOp51+(-23033065380660)*probOp41*probOp21*probOp51+30723113536560*probOp11*probOp41*probOp51+(-191866338399111114730)))/(1200 * ((3802*probOp41+2075505) * (73200*probOp11*probOp21+24260920*probOp11+(-18188370)*probOp21+3971767903)))



#R3: What is the expected cost per workflow execution?
#
#p51? [47.57, 52.97]
def property_R3_unknown_p51(list_samples):
    probOp51 = list_samples[0]
    return (-1 * (6056317028265807350*probOp51+(-59421709857035776147)))/(1121809240610595000)

#p41, p51? [43.307, 53.662]
def property_R3_unknown_p41p51(list_samples):
    probOp41 = list_samples[0]
    probOp51 = list_samples[1]
    return (417728878230300*probOp41*probOp51+(-13929532148789058550)*probOp51+(-7042438202780094606)*probOp41+138405806134373928671)/(4975328835000 * (26614*probOp41+518401))

#p21,p41,p51?
def property_R3_unknown_p21p41p51(list_samples):
    probOp21 = list_samples[0]
    probOp41 = list_samples[1]
    probOp51 = list_samples[2]
    return (354935363636700*probOp41*probOp21*probOp51+(-9844198299065578481)*probOp21+2785066420645751900*probOp51+4200045560299050*probOp21*probOp51+1451585969745307468*probOp41+(-215491645946442734)*probOp41*probOp21+(-154532848373400)*probOp41*probOp51+(-25712321567061670038))/(15000 * ((26614*probOp41+518401) * (302621*probOp21+(-66398242))))

#p11,p21,p41,p51?
def property_R3_unknown_p11p21p41p51(list_samples):
    probOp11 = list_samples[0]
    probOp21 = list_samples[1]
    probOp41 = list_samples[2]
    probOp51 = list_samples[3]
    return (2353585010400*probOp11*probOp41*probOp21*probOp51+473647437988413335100*probOp21+(-1906217679800436000)*probOp11*probOp21+(-266082712875191171600)*probOp11+(-128395812545798441085)*probOp51+(-194057411030267850)*probOp21*probOp51+258847347201120600*probOp11*probOp51+780993705726000*probOp11*probOp21*probOp51+(-83065947591868258676)*probOp41+739705865280494040*probOp41*probOp21+(-2976983057774400)*probOp11*probOp41*probOp21+(-712908698115028640)*probOp11*probOp41+(-135725119007634)*probOp41*probOp51+(-584807035459140)*probOp41*probOp21*probOp51+780056525280240*probOp11*probOp41*probOp51+1383182692773126363310)/(3000 * ((3802*probOp41+2075505) * (73200*probOp11*probOp21+24260920*probOp11+(-18188370)*probOp21+3971767903)))





# Prior knowledge of R1, as a Beta distribution with given parameters
#using Tensoflow distribution
dist_PK_R1 = tfd.Beta(tf.constant(470, dtype=tf.float64), tf.constant(30, dtype=tf.float64))
#using Scipy
# dist_PK_R1b = stats.beta(a=475, b=25)

dist_PK_R1b = stats.beta(a=47*200, b=3*200)


# Prior knowledge of R2, as a Gamma distribution with given parameters
#using Tensoflow distribution
dist_PK_R2 = tfd.Gamma(tf.constant(21*100, dtype=tf.float64), tf.constant(100.00, dtype=tf.float64))
#using Scipy
dist_PK_R2b = stats.gamma(scale=1/1000, a=21*1000)

dist_PK_R2b_withp4p5 = stats.gamma(scale=1/1000, a=20.374*1000)

dist_PK_R2b_withp2p4p5 = stats.gamma(scale=1/1000, a=20.04*1000)