

# import numpy as np
# from pymoo.problems import get_problem
# from pymoo.visualization.scatter import Scatter
# # The pareto front of a scaled zdt1 problem
# pf = get_problem("zdt1").pareto_front()
#
# # The result found by an algorithm
# A = pf[::10] * 1.2
#
# # plot the result
# Scatter(legend=True).add(pf, label="Pareto-front").add(A, label="Result").show()





def epsilon(front, reference_front):
    """
    :param front: list of lists for the computed front
    :param reference_front: the reference front
    :return: epsilon unary quality indicator
    """

    #number of objectives
    num_of_objectives = len(front[0])
    #epsilon value
    eps = - 1000
    #helper
    epsJ = 0
    epsK = 0

    for i in range(len(reference_front)):
        for j in range(len(front)):

            for k in range(num_of_objectives):
                epsTemp = front[j][k] - reference_front[i][k]
                if (k == 0):
                    epsK = epsTemp
                elif (epsK < epsTemp):
                    epsK = epsTemp

            if (j == 0):
                epsJ = epsK
            elif (epsJ > epsK):
                epsJ = epsK

        if (i == 0):
            eps = epsJ
        elif (eps < epsJ):
            eps = epsJ

    return eps


def calculate_indicators (data_dir, pareto_file_substring, reference_front_file, reference_point_file, NVARS, output_dir):
    from EPIK_Utils import get_files_using_substring, load_solutions_all
    from pymoo.indicators.igd import IGD as igd_pymoo
    from pymoo.indicators.hv import HV as HV_pymoo
    import pandas as pd
    import numpy as np
    import os

    pf_files    = get_files_using_substring(data_dir, pareto_file_substring)

    #load reference front
    df_reference        = load_solutions_all("", reference_front_file)
    reference_values    = df_reference.iloc[:,NVARS:]
    reference_values = np.array(reference_values)

    #load reference point (for HV)
    ref_point = load_solutions_all("", reference_point_file)
    ref_point = np.array(ref_point)[0, :]

    #init indicators
    igd = igd_pymoo(reference_values)
    hv = HV_pymoo(ref_point)

    #create dataframe to store results
    colHeaders = ['Filename', 'Epsilon', 'IGD', 'HV']
    df_indicators = pd.DataFrame(columns=colHeaders)

    for pf in pf_files:
        df_pf = load_solutions_all("", pf)
        pf_values = df_pf.iloc[:,NVARS:]

        pf_values        = np.array(pf_values)

        #Epsilon
        epsilon_value = epsilon(pf_values, reference_values)
        #IGD
        igd_value = igd(pf_values)
        #HV
        hv_value = hv(pf_values)

        #Append indicator values to dataframe
        df_indicators.loc[len(df_indicators)] = [pf, epsilon_value, igd_value, hv_value]

    #save_values_to_csv("Indicators.csv", df_indicators)
    filename = "Indicators.csv"
    filepath = os.path.join(output_dir, filename)
    df_indicators.to_csv(filepath, index=False, mode='w', header=True)

    print(filepath + " saved")

    return df_indicators



def prepare_indicators (PROBLEM_NAMES, data_dir="data"):
    import pandas as pd
    import os

    ALGORITHMS = ["NSGAII", "SPEA2", "CMAES"]
    INDICATORS = ["Epsilon", "IGD", "HV"]

    indicatorFiles = []


    for problem in PROBLEM_NAMES:
        problem_dir = os.path.join(data_dir, problem)

        filename = "Indicators.csv"
        filepath = os.path.join(problem_dir, filename)
        dfI = pd.read_csv(filepath)

        for indicator in INDICATORS:
            df = pd.DataFrame()

            for algo in ALGORITHMS:

                val = dfI[dfI['Filename'].str.contains(algo)][indicator].values
                df[algo] = val

            filename = problem +"_"+ indicator +".csv"
            filepath = os.path.join(problem_dir, filename)
            df.to_csv(filepath, sep='\t')
            indicatorFiles.append(filepath)
            print("Saved: ", filepath)

    return indicatorFiles


def prepare_data (indicatorFiles, data_dir="data"):
    import numpy as np
    import pandas as pd
    import os

    dfData = []

    for indicatorFile in indicatorFiles:
        # indicator_path = os.path.join(data_dir, indicatorFile)

        dff = pd.read_csv(indicatorFile, delimiter="\t", index_col=0)
        dfData.append(dff)

    eM = np.max([dfData[i].max() for i in (0, 3, 6)]) + 1
    em = np.min([dfData[i].min() for i in (0, 3, 6)]) - 1

    igdM = np.max([dfData[i].max() for i in (1, 4, 7)]) + 1
    igdm = np.min([dfData[i].min() for i in (1, 4, 7)]) - 1

    hvM = np.max([dfData[i].max() for i in  (2, 5, 8)]) + 10
    hvm = np.min([dfData[i].min() for i in (2, 5, 8)]) - 1

    maxMin = ([em, eM], [igdm, igdM], [hvm, hvM])

    return dfData, maxMin



if __name__ == '__main__':
    # ind = GD(pf)
    # print("GD", ind(A))
    #
    #
    # ind = IGD(pf)
    # print("IGD", ind(A))
    #
    #
    # ref_point = np.array([1.2, 1.2])
    # ind = HV(ref_point=ref_point)
    # print("HV", ind(A))


    # from EPIK_Utils import  load_solutions_all, get_files_using_substring
    # import numpy as np
    #
    # pf_files = get_files_using_substring("data", "ParetoFrontSet")
    # reference_pf = "data/Reference_Front.csv"
    #
    # NVARS = 2
    #
    # df_reference        = load_solutions_all(reference_pf)
    # reference_values    = df_reference.iloc[:,-NVARS:]
    #
    # for pf in pf_files:
    #     df_pf = load_solutions_all(pf)
    #     pf_values = df_pf.iloc[:,-NVARS:]
    #
    #     reference_values = np.array(reference_values)
    #     pf_values        = np.array(pf_values)
    #
    #
    #     #IGD
    #     from deap.benchmarks.tools import igd as igd_deap
    #     from pymoo.indicators.igd import IGD as igd_pymoo
    #     from pymoo.util import misc
    #
    #     igd = igd_pymoo(reference_values)
    #
    #     # pf_values = misc.at_least_2d_array(pf_values)
    #     # #Remove one dimension
    #     # igd.pf    = igd.pf[0,:,:]
    #     # pf_values = pf_values[0, :, :]
    #
    #     print("IGD: ", igd_deap(pf_values, reference_values), "\t", igd(pf_values))
    #
    #
    #     #Epsilon
    #     print("Epsilon: ", epsilon(pf_values, reference_values))
    #
    #
    #     #HV
    #     from pymoo.indicators.hv import HV as HV_pymoo
    #     from deap.benchmarks.tools import hypervolume as hv_deap
    #     from EPIK_Utils import load_solutions_all
    #
    #     ref_point = load_solutions_all("data/Reference_Point.csv")
    #
    #     ref_point = np.array(ref_point)[0,:]
    #     hv = HV_pymoo(ref_point)
    #     print("HV: ", hv(pf_values), "\n")


    # import os
    # data_dir = 'data'
    # substring = "ParetoFrontSet"
    # reference_front_file = os.path.join(data_dir, "Reference_Front.csv")
    # reference_point_file = os.path.join(data_dir, "Reference_Point.csv")
    # NVARS = 2
    # df = calculate_indicators(data_dir, substring, reference_front_file, reference_point_file, NVARS)
    #
    # print(df)


    '''
    Code for calculating the indicators based on the identified Pareto Front sets 
    and reference front/reference point 
    '''
    # paretoSet_dir               = "data/FX_R1R2_p2p4p5_100000/ParetoFrontSet"
    # substring                   = "ParetoFrontSet"
    # reference_front_filepath    = "data/FX_R1R2_p2p4p5_100000/Reference_Front.csv"
    # reference_point_filepath    = "data/FX_R1R2_p2p4p5_100000/Reference_Point.csv"
    # NVARS                       = 6
    # problem_dir                 = "data/FX_R1R2_p2p4p5_100000"
    # calculate_indicators(paretoSet_dir, substring, reference_front_filepath, reference_point_filepath, NVARS, problem_dir)



    #FX
    # filename = "fxBoxplots.pdf"
    # titles = ['p5', 'p4p5', 'p2p4p5']
    # PROBLEM_NAMES = ["FX_R1R2_p5_100000", "FX_R1R2_p4p5_100000", "FX_R1R2_p2p4p5_100000"]

    # #PRESTO
    filename = "plots/fprBoxplots.pdf"
    titles = ['p3', 'p2p3', 'p1p2p3']
    PROBLEM_NAMES = ["FPR_R1R2_p3", "FPR_R1R2_p2p3", "FPR_R1R2_p1p2p3"]



    indicatorFiles = prepare_indicators(PROBLEM_NAMES)

    dfData, maxMin = prepare_data(indicatorFiles)

    from EPIK_visualisation import do_boxplots_grid
    do_boxplots_grid(filename=filename, titles=titles, dfData=dfData, maxMin=maxMin)


