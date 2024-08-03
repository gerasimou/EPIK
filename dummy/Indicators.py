


def epsilon (front, reference_front):
    '''

    :param front: list of lists for the computed front
    :param reference_front: the reference front
    :return: epsilon unary quality indicator
    '''

    #number of objectives
    num_of_objectives = len(front[0])
    #epsilon value
    epsilon_value = - 1000
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
