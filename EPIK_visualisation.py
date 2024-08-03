import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def show_scatterPlot_Front__From_File(df_front_list, filename):
    for df_front in df_front_list:
        ax = sns.scatterplot(data=df_front, x=df_front.columns[-2], y=df_front.columns[-1])

    plt.legend(labels=['NSGAII', 'SPEA2'], title='Algorithm')
    plt.xlabel('R1')
    plt.ylabel('R2')

    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()



def show_scatterPlot_Front(first_front, population):
    # Find the number of columns
    colsNum = len(first_front[0].fitness.values)

    # Create the column headers
    colHeaders = ["P" + str(i + 1) for i in range(colsNum)]

    # Create value lists
    front_values = [front_element.fitness.values for front_element in first_front]
    population_values = [pop_element.fitness.values for pop_element in population]

    # Create dataframe
    df_front = pd.DataFrame(data=front_values, columns=colHeaders)
    df_population = pd.DataFrame(data=population_values, columns=colHeaders)

    sns.scatterplot(data=df_population, x=colHeaders[0], y=colHeaders[1])
    sns.scatterplot(data=df_front, x=colHeaders[0], y=colHeaders[1])

    plt.show()



def show_distPlot(list_a, list_b, list_property_eval_func, n, list_property_prior_knowledge, filename, labels):
    plots_num = len(list_property_eval_func)

    fig, axes = plt.subplots(1, plots_num, figsize=(plots_num * 5, 4))

    # Sample from the optimised dist. of parameters
    list_samples = [np.random.beta(list_a[i], list_b[i], size=n) for i in range(len(list_a))]

    for i in range(0, plots_num):
        # Sample from the PK
        samples_tar = list_property_prior_knowledge[i].rvs(n)

        # Calulate the sample of y via the sample of x
        sampled_Y = list_property_eval_func[i](list_samples)

        # Plot a histogram of the samples from the true and estimated distributions
        if (plots_num == 1):
            axis = axes
        else:
            axis = axes[i]
        sns.histplot(ax=axis, data=samples_tar, kde=True, label="PK")
        sns.histplot(ax=axis, data=sampled_Y, kde=True, label="Optimised")

    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()


def show_distPlot2(list_a, list_b, list_property_eval_func, n, list_property_prior_knowledge, filename, labels, title):
    plots_num = len(list_property_eval_func)

    # Sample from the optimised dist. of parameters
    list_samples = [np.random.beta(list_a[i], list_b[i], size=n) for i in range(len(list_a))]

    for i in range(0, plots_num):
        # Sample from the PK
        samples_tar = list_property_prior_knowledge[i].rvs(n)

        # Calulate the sample of y via the sample of x
        sampled_Y = list_property_eval_func[i](list_samples)

        sns.histplot(data=samples_tar, kde=True, label="PK")
        sns.histplot(data=sampled_Y, kde=True, label="Optimised")

        plt.legend(labels=['Domain expert', 'EPIK'], loc="upper right")
        plt.xlabel(labels[i])
        # plt.ylabel('R2')

        plt.title(title)

        plt.savefig(filename + labels[i] +".pdf", bbox_inches='tight', pad_inches=0)
        plt.show()



def show_distPlot_TFD(list_a, list_b, list_property_eval_func, n, list_property_prior_knowledge, filename, labels, title):
    plots_num = len(list_property_eval_func)

    # Sample from the optimised dist. of parameters
    list_samples = [np.random.beta(list_a[i], list_b[i], size=n) for i in range(len(list_a))]

    for i in range(0, plots_num):
        # Sample from the PK
        samples_tar = list_property_prior_knowledge[i].sample(n)

        # Calulate the sample of y via the sample of x
        sampled_Y = list_property_eval_func[i](list_samples)

        sns.histplot(data=samples_tar, kde=True, label="PK")
        sns.histplot(data=sampled_Y, kde=True, label="Optimised")

        plt.legend(labels=['Domain expert', 'EPIK'], loc="upper right")
        plt.xlabel(labels[i])

        plt.title(title)

        plt.savefig(filename + labels[i] +".pdf", bbox_inches='tight', pad_inches=0)
        plt.show()






def show_dist(best_found):
    import numpy as np
    from scipy.stats import beta
    import matplotlib.pyplot as plt

    from ReqsPRESTO import dist_PK_R1b,property_R1_unknown_p3

    fig, ax = plt.subplots(1, 1)

    n = 10000

    pior = [160, 40]

    samples_tar = dist_PK_R1b.rvs(n)
    sns.histplot(data=samples_tar, kde=True, label="PK")

    # best_found = [8.5929, 1.725]
    a = best_found[0]
    b = best_found[1]
    beta_vals = np.random.beta(a, b, n)
    # x = np.linspace(0, 1, 100)
    # x = np.arange(0, 1, 0.001)
    # beta_vals = beta.pdf(x, a, b)
    sampled_Y = property_R1_unknown_p3([beta_vals])
    # ax.plot(x, sampled_Y, 'r-', lw=5, alpha=0.6, label='beta pdf')
    # ax.hist(sampled_Y, density=True, color='r', histtype='step')
    sns.histplot(data=sampled_Y, kde=True, label="Optimised")

    # x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
    # ax.plot(x, beta.pdf(x, a, b), 'r-', lw=5, alpha=0.6, label='beta')


    ax.legend(loc='best', frameon=False)
    plt.show()



def do_boxplots_grid(filename, titles, dfData, maxMin, rows=3, cols = 3):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(rows, cols, figsize=(18, 10))

    # sns.set(font_scale=1.8)
    plt.xticks(fontsize=5)

    plt.subplots_adjust(left=0.3,
                          bottom=0.1,
                          right=0.9,
                          top=0.9,
                          wspace=0.05,
                          hspace=0.15)


    k = 0
    for j in range(int(len(dfData)/len(maxMin))):
        for i in range(int(len(dfData)/len(maxMin))):
            ax = sns.boxplot(ax=axes[i, j], data=dfData[k])
            # ax.set_ylim(maxMin[i])
            if i == 0:
                ax.set_title(titles[j])

            if i == 2:
                ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
            else:
                ax.axes.xaxis.set_ticklabels([])

            if j == 0:
                for label in (ax.get_yticklabels()):
                    label.set_fontsize(16)
            else:
                ax.axes.yaxis.set_ticklabels([])

            k += 1


    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()




def prepare_conformance_data (data_dir, conformance_files, colHeaders):
    dfData = pd.DataFrame(columns=colHeaders)

    for i in range(len(conformance_files)):
        #get filename
        file = conformance_files[i]

        #produce filepath
        filepath = os.path.join(data_dir, file)

        #read file.
        df = pd.read_csv(filepath)

        #extract optimisation values (the last column)
        values = df.iloc[:, -1]
        values = values[-30:]

        #add the extracted values to the dataframe
        dfData.iloc[:, i] = values

        #reset the index - just in case it is mixed up
        dfData.reset_index(drop=True, inplace=True)

    return dfData


def do_conformance_boxplots (data_dir, conformance_files, colHeaders):
    dfData = prepare_conformance_data(data_dir, conformance_files, colHeaders)

    plotName = "conformanceBoxplot1.pdf"

    #Prepare the data for producing a boxplot
    dfDataMelt = pd.melt(dfData)
    dfDataMelt.columns = ['Conformance Level', 'Distance']

    fig = plt.figure(figsize=(4, 6))
    sns.boxplot(data=dfDataMelt, x=dfDataMelt.columns[0], y=dfDataMelt.columns[1])
    plt.savefig(plotName, bbox_inches='tight', pad_inches=0.00)
    plt.savefig("conformanceBoxplot1.svg", bbox_inches='tight', pad_inches=0.02)
    plt.show()


def do_conformance_distPlots(data_dir, conformance_files, list_property_eval_func, n, list_property_prior_knowledge, outFilename, conformance_labels, property_labels):

    for i in range(len(conformance_files)):
        #get filename
        file = conformance_files[i]

        #produce filepath
        filepath = os.path.join(data_dir, file)

        #read file.
        df = pd.read_csv(filepath)

        #extract optimisation values (the last column)
        values = df.iloc[:, -1]
        valueMinIndex = np.argmin(values)

        dfMinEntry = df.loc[valueMinIndex, :]

        #     #extract the details of the best solution
        list_a = [dfMinEntry[0]]
        list_b = [dfMinEntry[1]]


        plots_num = len(list_property_eval_func)

        # n = n*5

        # Sample from the optimised dist. of parameters
        list_samples = [np.random.beta(list_a[i], list_b[i], size=n) for i in range(len(list_a))]

        for j in range(0, plots_num):
            # Sample from the PK
            samples_tar = list_property_prior_knowledge[j].sample(n)

            # Calulate the sample of y via the sample of x
            sampled_Y = list_property_eval_func[j](list_samples)

            sns.histplot(data=samples_tar, kde=True, label="PK")
            sns.histplot(data=sampled_Y, kde=True, label="Optimised")

            plt.legend(labels=['Domain expert', 'EPIK'], loc="upper right")
            plt.xlabel(property_labels[j], fontsize=14)
            plt.ylabel(ylabel='Count', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)


            # plt.title(title)

            filename = "FPR_Conformance_"+ str(conformance_labels[i]) +"_"+ property_labels[j] +".pdf"
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.show()



if __name__ == '__main__':
    from EPIK_Utils import load_solutions_all


    # # Pareto front
    # problem_name = "FPR_R1R2_p3"
    # filenames = [
    #     "ParetoFrontSetNSGAII-2.csv",  # 4,5,8,9,14,26/ 21,24,9
    #     "ParetoFrontSetSPEA2-7.csv",  # 7,12,14,19, 24,29 / 21,26
    #     # "ParetoFrontSetNSGAII-20.csv"
    # ]
    # data_dir = "data/" + problem_name + "/ParetoFrontSet"
    # outputPFfile = data_dir + "/" + problem_name + "_PF.pdf"
    #
    # # data_dir = "data/FPR_R1R2_p1p2p3/ParetoFrontSet"
    # # filenames = [
    # #                 "ParetoFrontSetNSGAII-9.csv",#4,5,8,9,14,26/ 21,24,9
    # #                 "ParetoFrontSetSPEA2-26.csv",#7,12,14,19, 24,29 / 21,26
    # #                 # "ParetoFrontSetNSGAII-20.csv"
    # # ]
    # # filename = data_dir + "/" + "FPR_R1R2_p1p2p3_PF.pdf"
    #
    # dfData = []
    # for filename in filenames:
    #     df = load_solutions_all(data_dir, filename)
    #     dfData.append(df)
    # show_scatterPlot_Front__From_File(dfData, outputPFfile)


    # #Dist plot
    # # from PRESTO_reqs import *
    # # from EPIK_Utils import Property_Dist
    # #
    # # # problem_name = "FPR_R1R2_p3"
    # # # n = 100000
    # # # list_property_eval_func = [property_R1_unknown_p3, property_R2_unknown_p3]
    # # # list_property_prior_knowledge = [dist_PK_R1b, dist_PK_R2b]
    # # # list_property_eval_distr = [Property_Dist.Beta, Property_Dist.Gamma]
    # # # data_dir = "data/" + problem_name +"/ParetoFrontSet"
    # # # filename = "ParetoFrontSetNSGAII-5.csv"
    # # # df = load_solutions_all(data_dir, filename)
    # # #
    # # # rowId = 1 #3,4,6,8
    # # # list_a = [df.iloc[rowId,i] for i in range(len(df.columns)) if "v" in df.columns[i] and i % 2 == 0]
    # # # list_b = [df.iloc[rowId,i] for i in range(len(df.columns)) if "v" in df.columns[i] and i % 2 == 1]
    # # #
    # # # outFilename = data_dir +"/"+ problem_name + "_Dist"
    # # # labels = ["R1", "R2"]
    # # # show_distPlot2(list_a, list_b, list_property_eval_func, n, list_property_prior_knowledge, outFilename, labels)
    # #
    # #
    # # # problem_name = "FPR_R1R2_p2p3"
    # # # n = 100000
    # # # list_property_eval_func = [property_R1_unknown_p3, property_R2_unknown_p3]
    # # # list_property_prior_knowledge = [dist_PK_R1b, dist_PK_R2b]
    # # # list_property_eval_distr = [Property_Dist.Beta, Property_Dist.Gamma]
    # # # data_dir = "data/" + problem_name +"/ParetoFrontSet"
    # # # filename = "ParetoFrontSetNSGAII-14.csv"
    # # # df = load_solutions_all(data_dir, filename)
    # # #
    # # # rowId = 4#3,4,6,8/ 1,4,5
    # # # list_a = [df.iloc[rowId,i] for i in range(len(df.columns)) if "v" in df.columns[i] and i % 2 == 0]
    # # # list_b = [df.iloc[rowId,i] for i in range(len(df.columns)) if "v" in df.columns[i] and i % 2 == 1]
    # # #
    # # # outFilename = data_dir +"/"+ problem_name + "_Dist"
    # # # labels = ["R1", "R2"]
    # # # show_distPlot2(list_a, list_b, list_property_eval_func, n, list_property_prior_knowledge, outFilename, labels)
    # #
    # #
    # # problem_name = "FPR_R1R2_p1p2p3"
    # # n = 100000
    # # list_property_eval_func = [property_R1_unknown_p1p2p3, property_R2_unknown_p1p2p3]
    # # list_property_prior_knowledge = [dist_PK_R1b, dist_PK_R2b]
    # # list_property_eval_distr = [Property_Dist.Beta, Property_Dist.Gamma]
    # # data_dir = "data/" + problem_name +"/ParetoFrontSet"
    # # filename = "ParetoFrontSetNSGAII-24.csv"
    # # df = load_solutions_all(data_dir, filename)
    # #
    # # rowId = 4#3,4,6,8/ 1,4,5
    # # list_a = [df.iloc[rowId,i] for i in range(len(df.columns)) if "v" in df.columns[i] and i % 2 == 0]
    # # list_b = [df.iloc[rowId,i] for i in range(len(df.columns)) if "v" in df.columns[i] and i % 2 == 1]
    # #
    # # outFilename = data_dir +"/"+ problem_name + "_Dist"
    # # labels = ["R1", "R2"]
    # # show_distPlot2  (list_a, list_b, list_property_eval_func, n, list_property_prior_knowledge, outFilename, labels)
    #
    #
    #
    # # import FX_reqs
    # # from EPIK_Utils import Property_Dist
    # # problem_name =  "FX_R2_p5b"
    # # n = 10000
    # # list_property_eval_func = [FX_reqs.property_R2_unknown_p51]
    # # list_property_prior_knowledge = [FX_reqs.dist_PK_R2b]
    # # list_property_eval_distr = [Property_Dist.Gamma]
    # #
    # # results = [0.6687269422365917,0.4132570695722175]
    # #
    # # # Prepare lists
    # # list_a = [results[i] for i in range(len(results)) if i % 2 == 0]
    # # list_b = [results[i] for i in range(len(results)) if i % 2 == 1]
    # #
    # # data_dir = "data"
    # # outFilename = data_dir +"/"+ problem_name +"_"
    # # labels = ["R1"]
    # # show_distPlot2  (list_a, list_b, list_property_eval_func, n, list_property_prior_knowledge, outFilename, labels)
    #
    #
    #
    # #Pareto front
    from EPIK_Utils import get_files_using_substring
    data_dir        = "data"
    problem_name    = "FX_R1R2_p4p5_100000"
    problem_dir     = os.path.join(data_dir, problem_name)
    pareto_dir      = os.path.join(data_dir, problem_name, "ParetoFrontSet")

    #filenames = get_files_using_substring(data_dir, "ParetoFrontSet")
    filenames = [
                    # "Reference_Front.csv",#7,12,14,19, 24,29 / 21,26
                    "ParetoFrontSetNSGAII-3.csv",  # 4,5,8,9,14,26/ 21,24,9
                    "ParetoFrontSetSPEA2-16.csv",
                    # "ParetoFrontSetNSGAII-10.csv",  # 4,5,8,9,14,26/ 21,24,9

    ]

    dfData = []
    for filename in filenames:
        df = load_solutions_all(pareto_dir, filename)
        dfData.append(df)
    filename = os.path.join(problem_dir, problem_name +".pdf")
    show_scatterPlot_Front__From_File(dfData, filename)





    # # from numpy.random import default_rng
    # #
    # # rng = default_rng(seed=111)
    # # rng.beta
    #
    # best_ind =   [7.84596931797008, 1.528311849559791]# [8.183254548852606, 1.5410445552854144]#[7.5801, 0.77523]
    #
    # from EPIK_visualisation import show_distPlot2
    # problem_name = "FPR"
    # list_a = [best_ind[i] for i in range(len(best_ind)) if i % 2 == 0]
    # list_b = [best_ind[i] for i in range(len(best_ind)) if i % 2 == 1]
    #
    # DATA_DIR = "data"
    # outFilename = DATA_DIR +"/"+ problem_name +"_"
    # labels = ["R1", "R2"]
    # title = problem_name
    # n = 10000
    #
    # from ReqsPRESTO import property_R1_unknown_p3, property_R2_unknown_p3, dist_PK_R1, dist_PK_R2
    # from EPIK_Utils import Property_Dist
    #
    # # list keeping the formulae of properties for which we have prior knowledge
    # list_property_eval_func = [property_R1_unknown_p3,  property_R2_unknown_p3]
    #
    # # list keeping the type of properties for which we have prior knowledge
    # # Property_Dist.Beta -> [0,1], Property_Dist.Gamma -> [0,\infty)
    # list_property_eval_distr = [Property_Dist.Beta, Property_Dist.Gamma]
    #
    # # list keeping the prior knowledge for the properties of interest
    # list_property_prior_knowledge = [dist_PK_R1, dist_PK_R2]
    #
    #
    # # show_distPlot2(list_a, list_b, list_property_eval_func, n, list_property_prior_knowledge,
    # #                outFilename, labels, title)
    #
    #
    # # show_dist(best_ind)
    #
    #
    # from EPIK_Utils import calculate_KL_divergence
    # kl_divergence = calculate_KL_divergence(list_a, list_b, n, list_property_eval_func,
    #                                         list_property_prior_knowledge, list_property_eval_distr)
    # print(kl_divergence)

