import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

#R1: "The system steady-state power utilisation"
# R{"power"}=? [ S ]
def property_R1_unknown_evTrans11(list_samples):
    evTrans11 = list_samples[0]
    res = (366568702513126894228142325823898692419897481388875498657026*evTrans11**2+5815789061554639445196049868811375043396505564944882008539663*evTrans11+12493053512041692602437800452598585981515248634549766840560430)/(500 * (398277656916667098254301119792493565304177688071445660918*evTrans11**2+6297724133945981189065989300509465668882620597781135920821*evTrans11+13293318648999333609309107380062651425619480845395887835674))
    return res


#R2: "The steady-state utilisation of the high-priority queue
#R{"QHLength"}=?[ S ]
def property_R2_unknown_evTrans11(list_samples):
    evTrans11 = list_samples[0]
    res =  (3 * (34688520703151686045324270599980808252575462612914123869282*evTrans11**2+539497828920354893272954645345313388378689805630206283438431*evTrans11+1036792048262208879418756114463290945519476427100405884970190))/(175 * (398277656916667098254301119792493565304177688071445660918*evTrans11**2+6297724133945981189065989300509465668882620597781135920821*evTrans11+13293318648999333609309107380062651425619480845395887835674))
    return res


#R3: "The steady-state utilisation of the low-priority queue
#R{"QLLength"}=?[ S ]
def property_R3_unknown_evTrans11(list_samples):
    evTrans11 = list_samples[0]
    res =  (105096121318343554038131919928370583291005979147222908352126*evTrans11**2+1658178393338373409609743329641057093608842250162455418000433*evTrans11+3459077160278813707086490907621214349673022901184849892908370)/(175 * (398277656916667098254301119792493565304177688071445660918*evTrans11**2+6297724133945981189065989300509465668882620597781135920821*evTrans11+13293318648999333609309107380062651425619480845395887835674))
    return res


#R4: "The number of requests lost at the steady state
#R{"TotalLost"}=? [ S ]
def property_R4_unknown_evTrans11(list_samples):
    evTrans11 = list_samples[0]
    res =  (111126049572243906603217487445394275827712 * (21164729048402808*evTrans11**2+328168602585955714*evTrans11+619302629044287135))/(125 * (398277656916667098254301119792493565304177688071445660918*evTrans11**2+6297724133945981189065989300509465668882620597781135920821*evTrans11+13293318648999333609309107380062651425619480845395887835674))
    return res


def property_R1_unknown_service (list_samples):
    service = list_samples[0]
    res = (512363607858981690731331644415416713811457517926678528000000000000000*service**15+593575086759273429780811377136205571993976244253324329779200*service**2+9892799033794889965894988360728066868872220960686125767424000*service**3+127125012147698645760278136976810915322233163932767299732670000*service**4+1329743386417672723214319143765760730528632597692973944453175000*service**5+11738422299225823469055983772283908916951226100491307986556421875*service**6+89289950053088709639911915474129226175581146692823568374738437500*service**7+593090532340830753585711652590483009977557236571416885307512500000*service**8+3448895927145496609431223018808872340237835981803955696321000000000*service**9+17402343390253314089426763667449264343531109016256400323500000000000*service**10+74772237318959546700256030676383637406177566126200540493900000000000*service**11+261138460292067538343261032447827848937283982115063719865000000000000*service**12+692089613078731217574025304744045967755141710452422434260000000000000*service**13+1205118638889570192288939801773418678623578626095715276800000000000000*service**14+25046773010598495973050644314699309991380428979000678809600*service+562724678125867300696866591781340165473101227680088457216)/(20 * (85225341757553322830938344145750282869419657607708672000000000000000*service**15+13804071785099382087925845979911757488232005680309868134400*service**2+230065093809183487578953217691350392299353975829909901568000*service**3+2956395631341828971169258999460718960982166603087611621690000*service**4+30924264800410993563123701017808389082061223202162184754725000*service**5+273573044370408138571720369988714504198786390757324333523265625*service**6+2096304141785394852592596514085330801092232915995554950095312500*service**7+14147439762561767214469055769121890984648127147827211117737500000*service**8+84890697932406153693166153468702044265453453191439523523000000000*service**9+451097087038045950203042539770258960281148721398329866500000000000*service**10+2093525924843832170621250608328882320197437137473342633700000000000*service**11+8244793503270894066897835247225320241416314406546645419000000000000*service**12+25991944998148040194392851639861514033664559487094643580000000000000*service**13+61015473915388028563685988860622072812364956226462156800000000000000*service**14+582483093269732464489549867783704883520475092534899507200*service+13086620421531797690624804460031166638909330876281126912))
    return res




# Prior knowledge of R1, as a Gamma distribution with given parameters
dist_PK_R1 = tfd.Gamma(tf.constant(184.26, dtype=tf.float64), tf.constant(100.0, dtype=tf.float64))

# Prior knowledge of R2, as a Gamma distribution with given parameters
dist_PK_R2 = tfd.Gamma(tf.constant(950.0, dtype=tf.float32), tf.constant(200.0, dtype=tf.float32))

# Prior knowledge of R3, as a Gamma distribution with given parameters
dist_PK_R3 = tfd.Gamma(tf.constant(950.0, dtype=tf.float32), tf.constant(200.0, dtype=tf.float32))

# Prior knowledge of R4, as a Gamma distribution with given parameters
dist_PK_R4 = tfd.Gamma(tf.constant(10, dtype=tf.float32), tf.constant(200.0, dtype=tf.float32))

dist_PK_R1_service = tfd.Gamma(tf.constant(4, dtype=tf.float64), tf.constant(10.0, dtype=tf.float64))


if __name__ == '__main__':
    import numpy as np
    list_a = [1.2399857900232094, 0.9317637101561115]#[3.75669219]
    list_b = [12.332129484179212, 41.59556694788443]#[1.85874307]
    n = 1000

    list_samples = [np.random.gamma(shape=list_a[i], scale=list_b[i], size=n) for i in range(len(list_a))]

    # # print(property_R2_unknown_evTrans11(list_samples))
    # # print(property_R3_unknown_evTrans11(list_samples))
    # # print(property_R4_unknown_evTrans11(list_samples))

    res_R1_service = [property_R1_unknown_service([list_samples[i]]) for i in range(len(list_samples))]


    import seaborn as sns
    import matplotlib.pyplot as plt
    ax = sns.histplot(data=dist_PK_R1_service.sample(n), kde=True, label="R1", color='y')
    for i in range(len(res_R1_service)):
        ax = sns.histplot(data=res_R1_service[i], kde=True, label="R1")
    plt.show()


    # evTrans11 = [np.arange(0.0, 100.0, 0.001)]
    # result_R1 = property_R1_unknown_evTrans11(evTrans11)
    # result_R2 = property_R2_unknown_evTrans11(evTrans11)
    # result_R3 = property_R3_unknown_evTrans11(evTrans11)
    # result_R4 = property_R1_unknown_service(evTrans11)
    #
    # sample_mean     = np.mean(result_R1)
    # sample_variance = np.var(result_R1)

    # dist1 = tfd.Gamma.experimental_from_mean_variance (mean=sample_mean, variance=sample_variance)

    from EPIK_Utils import kl_divergence_gamma
    # dist = kl_divergence_gamma(dist_PK_R1, dist_PK_R1)
    # print(dist)
    # dist = kl_divergence_gamma(dist1, dist_PK_R1)
    # tf
    # dist = tfp.distributions.kl_divergence(dist1, dist_PK_R1)
    # print(dist)

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # f, axs = plt.subplots(2, 2)
    #
    # ax = sns.histplot(data=result_R1, kde=True, label="R1", ax=axs[0,0])
    # ax = sns.histplot(data=result_R2, kde=True, label="R1", ax=axs[0,1])
    # ax = sns.histplot(data=result_R3, kde=True, label="R1", ax=axs[1,0])
    # ax = sns.histplot(data=result_R4, kde=True, label="R1", ax=axs[1,1])
    # # ax.set_ylim([0, 10000])
    # plt.show()


    # print(np.median(result_R1))