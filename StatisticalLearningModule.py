import copy
import pandas as pd
import numpy as np
from scipy.stats import poisson
import math


# -----------------
#  Missing value positioning part
# -----------------

def zinbem(data, fai, k):
    # fai Zero proportion
    # nu dropout size
    # k Expressed value
    # initialization

    data_t = data.T
    np.seterr(divide='ignore', invalid='ignore')
    m, n = data_t.shape
    nu = n * k
    iniresult = [fai, nu, k]

    def em(data, fai, nu, k):
        # e step
        dominator = 0
        meam_zero = 0
        for observation in data:
            num_zero = (observation == 0).sum()  # Number of zeros per line
            meam_zero += num_zero
            dominator_d = (1 - fai) * np.power((1 - k), nu)
            dominator_m = fai + (1 - fai) * np.power((1 - k), nu)
            dominator += dominator_d / dominator_m
        finaldominator = dominator / m

        # m step
        meam_zero = meam_zero / m
        new_param_fai = (n - 1 / (1 - finaldominator) * (n - meam_zero)) / n
        new_param_nu = n * finaldominator  # (n - n * new_param_fai) * finaldominator
        new_param_k = finaldominator
        return new_param_fai, new_param_nu, new_param_k

    statisparam = [[0, 0, 0]]
    tol = 0.1
    for itrm in range(50):
        new_param_fai, new_param_nu, new_param_k = em(data_t, fai, nu, k)
        delta_change = np.abs(new_param_k - k)
        statisparam.append([new_param_fai, np.round(new_param_nu), new_param_k])
        if delta_change < tol:
            break
        elif new_param_fai < 0:
            break
        elif np.round(new_param_nu) < 0 or new_param_nu > n:
            break
        elif new_param_k > 1 or new_param_k < 0:
            break
        else:
            fai = new_param_fai
            nu = np.round(new_param_nu)
            k = new_param_k

    getfaildparam = statisparam[-10:]

    result = 0
    for i in range(len(getfaildparam)):  # Take the last few
        if getfaildparam[i][1] == 0:
            continue
        elif getfaildparam[i][0] <= 0:
            continue
        elif getfaildparam[i][2] < 0.1 or getfaildparam[i][2] > 1:
            continue
        else:
            result = getfaildparam[i]

    if result == 0:
        return iniresult
    else:
        return result


def getparam(data, fai, k):
    m, n = data.shape
    # param initial
    np.seterr(divide='ignore', invalid='ignore')
    zinbemparm = zinbem(data, fai, k)
    inip = zinbemparm[1] / (zinbemparm[1] + n * zinbemparm[0])
    inibeta = zinbemparm[1]

    inizeta = 0
    realnumbersum = 0
    for item in data:
        lensum = len(item)
        lenzero = (item == 0).sum()
        itemvar = []
        # inixi+=np.mean(item)*(lensum/lenzero)

        mean = np.sum(item) / (lensum - lenzero)  # mean
        mean = np.nan_to_num(mean)
        realnumbersum += mean  # Cumulative mean
        if sum(item) == 0:
            continue
        for jtem in item:
            if jtem == 0:

                continue
            else:
                # inizeta+=np.power((jtem-inixi),2)/(lensum-lenzero)
                itemvar.append(jtem)

        itemvar = np.nan_to_num(itemvar)
        if sum(itemvar) == 0:
            thevar = 0
        else:
            thevar = np.var(itemvar)
            thevar = np.nan_to_num(thevar)

        inizeta += thevar
    inixi = realnumbersum / m
    inizeta = inizeta / m

    return inip, inibeta, inixi, inizeta


def calculateweight(data, fai, k):
    p, beta, xi, zeta = getparam(data, fai, k)
    proba_matrix = []
    zeronum = 0
    data_t = data.T
    for observation in data_t:
        zeronum += (observation == 0).sum()
        lensum = len(observation)
        lenzero = (observation == 0).sum()

        dropsum = np.round(beta / lenzero * 100 + beta)
        beta = np.round(beta)
        theta = lenzero / lensum

        dominator_d = theta * ((1 - p) * poisson.pmf(beta, dropsum))
        dominator_d = np.nan_to_num(dominator_d)
        dominator_m_a = theta * (p + (1 - p) * np.exp(-beta) + (1 - p) * (
            poisson.pmf(beta, dropsum)))
        dominator_m_a = np.nan_to_num(dominator_m_a)
        norsum = 0
        for i in observation:
            norsum += np.power((i - xi), 2)
            if norsum == 0:
                norsum=1
        if zeta == 0:
            zeta=1
        dominator_m_b = (1 - theta) * (1 / (math.sqrt(2 * math.pi) * zeta)) * np.exp(
            -(norsum) / (2 * np.power(zeta, 2)))
        dominator = dominator_d / (dominator_m_a + dominator_m_b)
        dominator = np.nan_to_num(dominator)
        proba_matrix.append(dominator)
    return proba_matrix


def distinguishdropoutevent(pd_data, inifai, inik, input_label, D_throd):  # input pd data

    def getTmax(pd_data, thord):
        thord = np.round(thord)
        targetcoordinate = []
        ar_data = np.asarray(pd_data)
        m, n = ar_data.shape
        for i in range(m):
            sort = np.sort(ar_data[i])  # Ascending order
            partiton = int(thord[i])
            targetvalue = sort[-partiton:]
            for j in range(n):
                if ar_data[i, j] in targetvalue:
                    targetcoordinate.append([i, j])
        return targetcoordinate

    data = copy.deepcopy(pd_data)
    proba_data = copy.deepcopy(pd_data)
    columns = proba_data.columns
    clu_label = input_label
    data['clu_label'] = clu_label  # Add label

    # Construction of probability matrix for subgroup data calculation
    for i in np.unique(clu_label):
        cludata = data[data.loc[:, 'clu_label'] == i]  # Cells labeled I

        ar_cludata = np.asarray(cludata)
        ar_cludata = ar_cludata[:, :-1]  # get data
        proba_list = calculateweight(ar_cludata, inifai, inik)  # Calculate probability
        proba_list = np.nan_to_num(proba_list)

        val = pd.Series(proba_list, index=columns)
        for j in cludata.index:
            proba_data.loc[j] = val

    ar_proba_data = np.asarray(proba_data)
    diffupsum = []
    diffmaxvalue = []
    for i in ar_proba_data:  # TODO
        if i.sum() == 0:
            diffupsum.append(0)
            diffmaxvalue.append(0)
        sort_matrix = np.sort(i, axis=0)
        # Calculate the position with the maximum probability difference
        diffmax = 0
        diffmaxindex = 0
        for j in range(len(i)):
            if (j + 1) == len(i):
                break
            elif sort_matrix[j + 1] == sort_matrix[j]:
                continue
            elif (sort_matrix[j + 1] - sort_matrix[j]) > diffmax:
                diffmax = sort_matrix[j + 1] - sort_matrix[j]
                diffmaxindex = j
            else:
                continue
        diffupsum.append(len(i) - diffmaxindex - 1)
        diffmaxvalue.append(diffmax)

    # Number of missing values threshold
    def findthrodindex(list, throd):
        throdindex = []
        sortlist = np.sort(list, axis=0)
        throdlist = sortlist[-throd:]
        for i in range(len(list)):
            if list[i] in throdlist:
                throdindex.append(i)
        return throdindex

    confidence_index = findthrodindex(diffmaxvalue, D_throd)
    meanconfidence = 0

    for i in range(len(confidence_index)):
        meanconfidence += diffupsum[confidence_index[i]]
    meanconfidence = meanconfidence / len(confidence_index)

    mu = 1
    for index in range(len(diffupsum)):
        if meanconfidence == 0:
            print("the meanconfidence==0")
            break
        elif index in confidence_index:
            diffupsum[index] = diffupsum[index]
        else:
            if meanconfidence <= diffupsum[index]:
                diffupsum[index] = diffupsum[index] + meanconfidence * (meanconfidence - diffupsum[index]) / \
                                   diffupsum[index] * mu
            else:
                diffupsum[index] = diffupsum[index] + diffupsum[index] * (
                        (meanconfidence - diffupsum[index]) / meanconfidence) * mu

    dropoutcoordinate = getTmax(proba_data, np.round(diffupsum))
    return dropoutcoordinate
