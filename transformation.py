import pandas as pd
import numpy as np
import re
from scipy.stats import weibull_min

def saturation_hill(
    x
    ,alpha
    ,gamma
    ,x_marginal = None):
    # stopifnot(length(alpha) == 1)
    assert len([alpha]) == 1, "alpha must be a single value"
    # stopifnot(length(gamma) == 1)
    assert len([gamma]) == 1, "gamma must be a single value"
    inflexion = np.dot([np.min(x),np.max(x)], np.array([1 - gamma, gamma]))
    # inflexion = c(range(x) %*% c(1 - gamma, gamma)) # linear interpolation by dot product
    if x_marginal is None:
        x_scurve = x**alpha / (x**alpha + inflexion**alpha) # plot(x_scurve) summary(x_scurve)
    else:
        x_scurve = x_marginal**alpha / (x_marginal**alpha + inflexion**alpha)
    return x_scurve

def adstock_geometric(
    x
    ,theta
    ):
    # stopifnot(length(theta) == 1)
    assert len([theta]) == 1, "theta must be a single value"
    if len(x) > 1:
        x_decayed = [x[0]] + [0]*(len(x)-1)
        
        for xi in range(2,len(x_decayed)):
            x_decayed[xi] = x[xi] + theta * x_decayed[xi - 1]
        
        thetaVecCum = [theta]
        for t in range(len(x)):
            # thetaVecCum[t] = thetaVecCum[t - 1] * theta
            thetaVecCum.append(thetaVecCum[t] * theta)
    else:
        x_decayed = x
        thetaVecCum = theta
    inflation_total = sum(x_decayed) / sum(x)
    return {
        "x" : x
        ,"x_decayed" : x_decayed
        ,"thetaVecCum" : thetaVecCum
        ,"inflation_total" : inflation_total
        }

def adstock_weibull(
    x
    ,shape
    ,scale
    ,windlen = None
    ,type = "cdf"
    ):
    if windlen is None:
        windlen = len(x)
    # stopifnot(length(shape) == 1)
    assert len(shape) == 1, "shape must be a single value"
    # stopifnot(length(scale) == 1)
    assert len(scale) == 1, "scale must be a single value"
    if len(x) > 1:
        # check_opts(tolower(type), c("cdf", "pdf"))
        x_bin = list(range(1, windlen+1))
        scaleTrans = np.round(np.quantile(np.arange(1, windlen+1), scale))
        if shape == 0 or scale == 0:
            x_decayed = x
            thetaVecCum = thetaVec = [0] * windlen
        else:
            if "cdf" in type.lower():
                thetaVec = np.concatenate(([1], 1 - weibull_min.cdf(x_bin[:-1], shape, scale=scaleTrans))) # plot(thetaVec)
                thetaVecCum = np.cumprod(thetaVec) # plot(thetaVecCum)
            elif "pdf" in type.lower():
                thetaVecCum = self.normalize(weibull_min.pdf(x_bin, shape, scale=scaleTrans)) # plot(thetaVecCum)

            x_decayed = np.zeros(windlen)
            for i in range(len(x)):
                x_vec = np.concatenate((np.zeros(i), np.full(windlen - i, x[i])))
                thetaVecCumLag = np.concatenate((np.zeros(i), thetaVecCum[:-i]))
                x_prod = x_vec * thetaVecCumLag
                x_decayed += x_prod

            x_decayed = np.sum(x_decayed, axis=0)[np.arange(len(x))]
    else:
        x_decayed = x
        thetaVecCum = 1
    
    inflation_total = sum(x_decayed) / sum(x)

    return {
        "x" : x
        ,"x_decayed" : x_decayed
        ,"thetaVecCum" : thetaVecCum
        ,"inflation_total" : inflation_total
        }

def normalize(x):
    if (max(x) - min(x) == 0):
        return [1] + [0] * (len(x) - 1)
    else:
        return (x - min(x)) / (max(x) - min(x))
    
def transform_adstock(
    InputCollect
    ,x 
    ,theta = None
    ,shape = None
    ,scale = None
    ,windlen = None
    ):
    if windlen is None:
        windlen = len(x)
    # check_adstock(adstock)
    if InputCollect["adstock"][0] == "geometric":
        x_list_sim = adstock_geometric(x = x, theta = theta)
    elif InputCollect["adstock"][0] == "weibull_cdf":
        x_list_sim = adstock_weibull(x = x, shape = shape, scale = scale, windlen = windlen, type = "cdf")
    elif InputCollect["adstock"][0] == "weibull_pdf":
        x_list_sim = adstock_weibull(x = x, shape = shape, scale = scale, windlen = windlen, type = "pdf")
    return x_list_sim

def run_transformations(InputCollect,hypParamSam):
    all_media = InputCollect["all_media"]
    rollingWindowStartWhich = InputCollect["rollingWindowStartWhich"][0]
    rollingWindowEndWhich = InputCollect["rollingWindowEndWhich"][0]
    dt_modAdstocked = pd.DataFrame.from_dict(InputCollect["dt_mod"]).drop(["ds"],axis=1)

    mediaAdstocked = pd.DataFrame()
    mediaImmediate = pd.DataFrame()
    mediaCarryover = pd.DataFrame()
    mediaVecCum = pd.DataFrame()
    mediaSaturated = pd.DataFrame()
    mediaSaturatedImmediate = pd.DataFrame()
    mediaSaturatedCarryover = pd.DataFrame()

    for v in range(len(all_media)):
        ################################################
        ## 1. Adstocking (whole data)
        print("## 1. Adstocking (whole data)")
        m = dt_modAdstocked[all_media[v]]
        if InputCollect["adstock"][0] == "geometric":
            theta = hypParamSam[f"{all_media[v]}_thetas"].values[0]#[0][0]
            x_list = transform_adstock(InputCollect, m, theta = theta)
        if re.search("weibull", InputCollect["adstock"][0]) is not None:
            shape = hypParamSam[f"{all_media[v]}_shapes"].values[0] #[0][0]
            scale = hypParamSam[f"{all_media[v]}_scales"].values[0] #[0][0]
            x_list = transform_adstock(InputCollect, m, shape = shape, scale = scale)
        m_adstocked = {all_media[v]:x_list["x_decayed"]}
        mediaAdstocked[all_media[v]] = m_adstocked
        m_carryover = m_adstocked[all_media[v]] - m
   
        m = pd.DataFrame(m)
        m_carryover = pd.DataFrame(m_carryover)
        m_adstocked = pd.DataFrame(m_adstocked)

        negative_indices = m_carryover[m_carryover[all_media[v]] < 0].index
        m.loc[negative_indices, all_media[v]] = m_adstocked.loc[negative_indices, all_media[v]]
        # m.loc[m_carryover < 0] = m_adstocked.loc[m_carryover < 0] # adapt for weibull_pdf with lags
        m_carryover.loc[m_carryover[all_media[v]] < 0,all_media[v]] = 0 # adapt for weibull_pdf with lags
        mediaImmediate = pd.concat([mediaImmediate,m],axis=1)
        mediaCarryover = pd.concat([mediaCarryover,m_carryover],axis=1)
        mediaVecCum = pd.concat([mediaVecCum,pd.DataFrame({all_media[v]:x_list["thetaVecCum"]})],axis=1)

        ################################################
        ## 2. Saturation (only window data)
        print("## 2. Saturation (only window data)")
        m_adstockedRollWind = m_adstocked.loc[rollingWindowStartWhich:rollingWindowEndWhich+1]
        m_carryoverRollWind = m_carryover.loc[rollingWindowStartWhich:rollingWindowEndWhich+1]

        alpha = hypParamSam[f"{all_media[v]}_alphas"].values[0]#[0]
        gamma = hypParamSam[f"{all_media[v]}_gammas"].values[0]#[0]
        mediaSaturated[all_media[v]] = saturation_hill(
            m_adstockedRollWind
            ,alpha = alpha
            , gamma = gamma
            )
        mediaSaturatedCarryover[all_media[v]] = saturation_hill(
            m_adstockedRollWind
            ,alpha = alpha
            ,gamma = gamma
            ,x_marginal = m_carryoverRollWind
            )
        mediaSaturatedImmediate[all_media[v]] = mediaSaturated[all_media[v]] - mediaSaturatedCarryover[all_media[v]]

    mediaSaturated = pd.DataFrame(mediaSaturated)
    
    # mediaSaturatedImmediate 
    dt_saturatedImmediate = pd.DataFrame(mediaSaturatedImmediate)
    # mediaSaturatedCarryover 
    dt_saturatedCarryover = pd.DataFrame(mediaSaturatedCarryover)

    # mediaAdstocked.columns = mediaImmediate.columns = mediaCarryover.columns = mediaVecCum.columns = \
    # mediaSaturated.columns = dt_saturatedImmediate.columns = dt_saturatedCarryover.columns = all_media
    
    dt_modAdstocked.drop(columns=all_media, inplace=True)
    dt_modAdstocked = pd.concat([dt_modAdstocked, mediaAdstocked], axis=1)

    dt_modSaturated = dt_modAdstocked.iloc[rollingWindowStartWhich:rollingWindowEndWhich+1] \
                .drop(all_media, axis=1) \
                .join(mediaSaturated)

    # dt_saturatedImmediate = pd.concat(mediaSaturatedImmediate, axis=1)
    dt_saturatedImmediate[dt_saturatedImmediate is None] = 0
    # dt_saturatedCarryover = pd.concat(mediaSaturatedCarryover, axis=1)
    dt_saturatedCarryover[dt_saturatedCarryover is None] = 0
    
    return {
        "dt_modSaturated" : dt_modSaturated,
        "dt_saturatedImmediate" : dt_saturatedImmediate,
        "dt_saturatedCarryover" : dt_saturatedCarryover
    }