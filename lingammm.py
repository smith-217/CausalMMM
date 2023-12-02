import pandas as pd
import numpy as np
import warnings
import os
import re
import CausalMMM.old.checks as ck
from datetime import datetime as dt
# import LiNGAMMM.refresh as ref
import hyper_params as hp
import nevergrad as ng
import math
from tqdm import tqdm
import time
from scipy.stats import uniform
import lingam_model as lm
import transformation as trs
# from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import scale
from sklearn.linear_model import Ridge
import json
warnings.simplefilter('ignore')

def lambda_seq(self,x,y,seq_len = 100,lambda_min_ratio = 0.0001):
    def mysd(y):
        return np.sqrt(np.sum((y - np.mean(y)) ** 2) / len(y))
    
    mysd_val = x.apply(mysd,axis=0)

    sx = (x - np.mean(x, axis=0)) / mysd_val

    check_nan = np.apply_along_axis(lambda sxj: np.all(np.isnan(sxj)), axis=0, arr=sx)
    na_colnum = [s for s in range(len(check_nan)) if check_nan[s]==True]

    sx.iloc[:,na_colnum].fillna(0,inplace=True)
    sy = y

    # 0.001 is the default smalles alpha value of glmnet for ridge (alpha = 0)
    sx_tmp = sx.apply(lambda x: x*sy)
    lambda_max = max(abs(np.sum(sx_tmp, axis=0))) / (0.001 * x.shape[0])
    # lambda_max_log = math.log(lambda_max)
    # log_step = (np.log(lambda_max) - np.log(lambda_max * lambda_min_ratio)) / (seq_len - 1)
    log_seq = np.linspace(np.log(lambda_max), np.log(lambda_max * lambda_min_ratio), num=seq_len)
    lambdas = np.exp(log_seq)
    return lambdas

def model_decomp(
        InputCollect,
        dt_modSaturated,
        dt_saturatedImmediate,
        dt_saturatedCarryover,
        mod_output
):
    ## Input for decomp
    y = dt_modSaturated["dep_var"]
    x = dt_modSaturated.drop("dep_var",axis=1)
    intercept = mod_output["df_int"]
    x_name = [s for s in x.columns]
    x_factor = [name for name in x_name if isinstance(name, str)]

    ## Decomp x
    xDecomp = pd.DataFrame()
    for each_col in mod_output["coef_df"].columns:
        xDecomp[each_col] = x[each_col] * mod_output["coef_df"][each_col][0]
    # xDecomp = pd.DataFrame({f"regressor_{i}": regressor * coeff for i, (regressor, coeff) in enumerate(zip(x, mod_output["coefs"][1:]), start=1)})
    xDecomp = pd.concat([pd.DataFrame({"intercept": [intercept] * xDecomp.shape[0]}), xDecomp], axis=1)
    xDecompOut = pd.concat([InputCollect["dt_modRollWind"]["ds"].reset_index(drop=True), pd.DataFrame({"y":y}), pd.DataFrame({"y_pred":mod_output["y_pred"]}), xDecomp], axis=1)

    ## Decomp immediate & carryover response
    sel_coef = [col for col in mod_output["coef_df"].columns if col in dt_saturatedImmediate.columns]
    coefs_media = mod_output["coef_df"][sel_coef]
    mediaDecompImmediate = pd.DataFrame()
    mediaDecompCarryover = pd.DataFrame()
    for each_col in sel_coef:
        mediaDecompImmediate = pd.concat([mediaDecompImmediate
                                            ,pd.DataFrame({each_col:dt_saturatedImmediate[each_col].reset_index(drop=True) * coefs_media[each_col].values[0]})
                                            ],axis=1)
        mediaDecompCarryover = pd.concat([mediaDecompCarryover
                                            ,pd.DataFrame({each_col:dt_saturatedCarryover[each_col].reset_index(drop=True) * coefs_media[each_col].values[0]})
                                            ],axis=1)
    ## Output decomp
    y_hat = xDecomp.sum(axis=1,skipna=True)
    y_hat_scaled = xDecomp.abs().sum(axis=1) #np.nansum(np.abs(xDecomp), axis=1)
    xDecompOutPerc_scaled = xDecomp.abs() / y_hat_scaled
    xDecompOut_scaled = y_hat * xDecompOutPerc_scaled

    temp = xDecompOut[["intercept"] + x_name]
    xDecompOutAgg = pd.DataFrame(temp.sum(axis=0)).reset_index()
    xDecompOutAgg.columns = ["rn","xDecompAgg"]

    xDecompOutAggPerc = pd.DataFrame({"rn":[s for s in temp.columns],"xDecompPerc":xDecompOutAgg["xDecompAgg"] / y_hat.sum()})

    xDecompOutAggMeanNon0 = [np.mean(x[x != 0]) if np.mean(x[x > 0]) else 0 for _, x in temp.iteritems()]
    xDecompOutAggMeanNon0_list = [0 if np.isnan(x) else x for x in xDecompOutAggMeanNon0]
    xDecompOutAggMeanNon0_col = [s for s, x in temp.iteritems()]
    xDecompOutAggMeanNon0 = pd.DataFrame({"rn":xDecompOutAggMeanNon0_col,"xDecompMeanNon0":xDecompOutAggMeanNon0_list})
    
    xDecompOutAggMeanNon0Perc_list = xDecompOutAggMeanNon0["xDecompMeanNon0"] / sum(xDecompOutAggMeanNon0["xDecompMeanNon0"])
    xDecompOutAggMeanNon0Perc = pd.DataFrame({"rn":xDecompOutAggMeanNon0_col,"xDecompMeanNon0Perc":xDecompOutAggMeanNon0Perc_list})

    refreshAddedStartWhich = [s for s in xDecompOut[xDecompOut["ds"] == InputCollect["refreshAddedStart"]].index][0]
    refreshAddedEnd = xDecompOut["ds"].max()
    refreshAddedEndWhich = [s for s in xDecompOut[xDecompOut["ds"] == refreshAddedEnd].index][-1]

    temp = xDecompOut.loc[xDecompOut["ds"].between(InputCollect["refreshAddedStart"], refreshAddedEnd),["intercept"] + x_name]
    xDecompOutAggRF_list = [sum(x) for _, x in temp.iteritems()]
    xDecompOutAggRF_col = [_ for _, x in temp.iteritems()]
    xDecompOutAggRF = pd.DataFrame({"rn":xDecompOutAggRF_col,"xDecompAggRF":xDecompOutAggRF_list})
    
    y_hatRF = y_hat.loc[refreshAddedStartWhich:refreshAddedEndWhich]

    xDecompOutAggPercRF_list = xDecompOutAggRF["xDecompAggRF"] / y_hatRF.sum()
    xDecompOutAggPercRF = pd.DataFrame({"rn":xDecompOutAggRF["rn"],"xDecompPercRF":xDecompOutAggPercRF_list})

    xDecompOutAggMeanNon0RF = [np.mean(x[x != 0]) if np.mean(x[x > 0]) == np.mean(x[x > 0]) else 0 for x in temp.values.T]
    xDecompOutAggMeanNon0RF_list = np.where(np.isnan(xDecompOutAggMeanNon0RF), 0, xDecompOutAggMeanNon0RF)
    xDecompOutAggMeanNon0RF_col = [s for s in temp.columns]
    xDecompOutAggMeanNon0RF = pd.DataFrame({"rn":xDecompOutAggMeanNon0RF_col,"xDecompMeanNon0RF":xDecompOutAggMeanNon0RF_list})

    xDecompOutAggMeanNon0PercRF_list = xDecompOutAggMeanNon0RF["xDecompMeanNon0RF"] / sum(xDecompOutAggMeanNon0RF["xDecompMeanNon0RF"])
    xDecompOutAggMeanNon0PercRF = pd.DataFrame({"rn":xDecompOutAggMeanNon0RF["rn"],"xDecompMeanNon0PercRF":xDecompOutAggMeanNon0PercRF_list})

    coefsOutCat = coefsOut = pd.DataFrame({"rn": mod_output["coef_df"].columns, "coefs": mod_output["coef_df"].values.flatten()})
    if len(x_factor) > 0:
        for factor in x_factor:
            coefsOut["rn"] = coefsOut["rn"].str.replace(f"{factor}.*", factor)

    rn_order = [s for s in xDecompOutAgg.keys()]
    rn_order = [s.replace("intercept", "(Intercept)") for s in rn_order]

    coefsOut = coefsOut.groupby(coefsOut["rn"]).mean()
    coefsOut = coefsOut.rename(columns={coefsOut.columns[0]: "coef"})
    coefsOut = coefsOut.reset_index()
    coefsOut = coefsOut.sort_values("rn", key=lambda x: x.map(dict(zip(rn_order, range(len(rn_order)))))).reset_index(drop=True)

    pos_df = pd.DataFrame({"rn":xDecompOutAgg["rn"],"pos":xDecompOutAgg["xDecompAgg"] >= 0})

    decompOutAgg = coefsOut.merge(xDecompOutAgg,on="rn",how="outer")
    decompOutAgg = decompOutAgg.merge(xDecompOutAggPerc,on="rn",how="outer")
    decompOutAgg = decompOutAgg.merge(xDecompOutAggMeanNon0,on="rn",how="outer")
    decompOutAgg = decompOutAgg.merge(xDecompOutAggMeanNon0Perc,on="rn",how="outer")
    decompOutAgg = decompOutAgg.merge(xDecompOutAggRF,on="rn",how="outer")
    decompOutAgg = decompOutAgg.merge(xDecompOutAggPercRF,on="rn",how="outer")
    decompOutAgg = decompOutAgg.merge(xDecompOutAggMeanNon0RF,on="rn",how="outer")
    decompOutAgg = decompOutAgg.merge(xDecompOutAggMeanNon0PercRF,on="rn",how="outer")
    decompOutAgg = decompOutAgg.merge(pos_df,on="rn",how="outer")
    
    dec_out = {}
    dec_out["xDecompVec"] = xDecompOut
    dec_out["xDecompVec_scaled"] = xDecompOut_scaled
    dec_out["xDecompAgg"] = decompOutAgg
    dec_out["coefsOutCat"] = coefsOutCat
    dec_out["mediaDecompImmediate"] = mediaDecompImmediate = pd.concat([xDecompOut["ds"],mediaDecompImmediate,xDecompOut["y"]],axis=1)
    dec_out["mediaDecompCarryover"] = pd.concat([xDecompOut["ds"],mediaDecompCarryover,xDecompOut["y"]],axis=1)
    return dec_out

# Must remain within this function for it to work
def robyn_iterations(
        InputCollect,
        i,
        hypParamSam,
        hyper_fixed,
        lambda_min,
        lambda_max,
        lambda_min_ratio,
        trial,
        add_penalty_factor,
        ts_validation,
        dt_spendShare,
        refresh,
        refresh_steps,
        rssd_zero_penalty,
        t0
):
    t1 = dt.now()

    #### Transform media for model fitting
    temp = trs.run_transformations(InputCollect,hypParamSam)
    
    dt_modSaturated = temp["dt_modSaturated"]
    dt_saturatedImmediate = temp["dt_saturatedImmediate"]
    dt_saturatedCarryover = temp["dt_saturatedCarryover"]

    #####################################
    #### Split train & test and prepare data for modelling

    dt_window = dt_modSaturated.copy()

    ## Contrast matrix because glmnet does not treat categorical variables (one hot encoding)
    y_window = dt_window["dep_var"]
    x_window = pd.get_dummies(dt_window.drop("dep_var",axis=1))
    y_train = y_val = y_test = y_window
    x_train = x_val = x_test = x_window

    ## Split train, test, and validation sets
    train_size = hypParamSam["train_size"].iloc[-1]#[0]
    val_size = test_size = (1 - train_size) / 2
    if train_size < 1:
        train_size_index = int(np.floor(np.quantile(np.arange(dt_window.shape[0]), train_size)))
        val_size_index = train_size_index + round(val_size * dt_window.shape[0])
        y_train = y_window.iloc[1:train_size_index]
        y_val = y_window.iloc[(train_size_index + 1):val_size_index]
        y_test = y_window.iloc[(val_size_index + 1):y_window.shape[0]]
        x_train = x_window.iloc[1:train_size_index, ]
        x_val = x_window.iloc[(train_size_index + 1):val_size_index, ]
        x_test = x_window.iloc[(val_size_index + 1):y_window.shape[0], ]
    else:
        y_val = y_test = x_val = x_test = None
        
    ## Define and set sign control
    dt_sign = dt_window.drop("dep_var",axis=1)
    
    if not isinstance(InputCollect["prophet_signs"],list): prophet_signs = [InputCollect["prophet_signs"]]
    else: prophet_signs = InputCollect["prophet_signs"]

    if not isinstance(InputCollect["context_signs"],list): context_signs = [InputCollect["context_signs"]]
    else: context_signs = InputCollect["context_signs"]

    if not isinstance(InputCollect["paid_media_signs"],list): paid_media_signs = [InputCollect["paid_media_signs"]]
    else: paid_media_signs = InputCollect["paid_media_signs"]

    if not isinstance(InputCollect["organic_signs"],list): organic_signs = [InputCollect["organic_signs"]]
    else: organic_signs = InputCollect["organic_signs"]

    if not isinstance(InputCollect["prophet_vars"],list): prophet_vars = [InputCollect["prophet_vars"]]
    else: prophet_vars = InputCollect["prophet_vars"]
    
    if not isinstance(InputCollect["context_vars"],list): context_vars = [InputCollect["context_vars"]]
    else: context_vars = InputCollect["context_vars"]

    if not isinstance(InputCollect["paid_media_spends"],list): paid_media_spends = [InputCollect["paid_media_spends"]]
    else: paid_media_spends = InputCollect["paid_media_spends"]

    if not isinstance(InputCollect["organic_vars"],list): organic_vars = [InputCollect["organic_vars"]]
    else: organic_vars = InputCollect["organic_vars"]

    x_sign_vals = prophet_signs+context_signs+paid_media_signs+organic_signs
    x_sign_vals = [s for s in x_sign_vals if s!=None]
    x_sign_cols = prophet_vars + context_vars + paid_media_spends + organic_vars
    x_sign_cols = [s for s in x_sign_cols if s!=None]
    x_sign = {}
    for key,val in zip(x_sign_cols,x_sign_vals): x_sign[key] = val

    check_factor = dt_sign.dtypes.apply(lambda x: True if pd.api.types.is_categorical_dtype(x) else False)

    lower_limits = upper_limits = None
    check_factor_cols = [s for s in check_factor.keys()]
    x_sign_cols = [s for s in x_sign.keys()]
    # print(dt_sign.tail(3))
    for s in range(len(check_factor)):
        if check_factor[check_factor_cols[s]] == True:
            level_n = len(dt_sign.iloc[:, s].astype(str).drop_duplicates())
            if level_n <= 1: raise ValueError("All factor variables must have more than 1 level")
            if x_sign[x_sign_cols[s]] == "positive": lower_vec = [0] * (level_n - 1)
            else: lower_vec = [-float('inf')] * (level_n - 1)
            if x_sign[x_sign_cols[s]] == "negative": upper_vec = [0] * (level_n - 1)
            else: upper_vec = [float("inf")] * (level_n - 1)
            lower_limits = lower_limits+lower_vec
            upper_limits = upper_limits+upper_vec
        else:
            lower_limits = np.where(x_sign[x_sign_cols[s]] == "positive", 0, -np.inf)
            upper_limits = np.append(upper_limits, [0 if x_sign[x_sign_cols[s]] == "negative" else np.inf])

    #####################################
    #### Fit ridge regression with nevergrad's lambda
    lambda_hp = hypParamSam['lambda'] #self.hypParamSamNG['lambda']#[i]

    if hyper_fixed == False: lambda_scaled = lambda_min + (lambda_max - lambda_min) * lambda_hp
    else: lambda_scaled = lambda_hp

    if add_penalty_factor: penalty_factor = hypParamSam.loc[i, np.grep("_penalty", hypParamSam.columns)].tolist()

    else: penalty_factor = np.ones(x_train.shape[1])

    #####################################
    ## NRMSE: Model's fit error
    if InputCollect["prior_knowledge"] is not None:
        mod_output = lm.causal_prediction(x_train, y_train,x_val, y_val,x_test, y_test,
                                          lambda_scaled = lambda_scaled,prior_knowledge = InputCollect["prior_knowledge"])
    else:
        mod_output = lm.causal_prediction(x_train, y_train,x_val, y_val,x_test, y_test,lambda_scaled = lambda_scaled)

    decomp_collect = model_decomp(
        InputCollect,
        dt_modSaturated,
        dt_saturatedImmediate,
        dt_saturatedCarryover,
        mod_output
    )

    nrmse = np.where(ts_validation, mod_output["nrmse_val"], mod_output["nrmse_train"])
    mape = 0
    # df_int = mod_output["df_int"] #self.df_int

    #####################################
    #### MAPE: Calibration error
    if InputCollect["calibration_input"] is not None:
        liftCollect = None
        warnings.warn("Caliburation modeling is currently not available...")
        # liftCollect = robyn_calibrate(
        # calibration_input = calibration_input,
        # df_raw = dt_mod,
        # hypParamSam = hypParamSam,
        # wind_start = rollingWindowStartWhich,
        # wind_end = rollingWindowEndWhich,
        # dayInterval = InputCollect$dayInterval,
        # dt_modAdstocked = InputCollect$dt_mod,
        # adstock = adstock,
        # xDecompVec = decompCollect$xDecompVec,
        # coefs = decompCollect$coefsOutCat
        # )
        # mape = mean(liftCollect$mape_lift, na.rm = True)

    #####################################
    #### DECOMP.RSSD: Business error
    # Sum of squared distance between decomp share and spend share to be minimized
    dt_decompSpendDist_dt = decomp_collect["xDecompAgg"][["rn"]]
    dt_decompSpendDist_dt = dt_decompSpendDist_dt[dt_decompSpendDist_dt["rn"].isin(InputCollect["paid_media_spends"])]

    temp = decomp_collect["xDecompAgg"][["rn","xDecompPerc"]]
    effect_share = temp[temp["rn"].isin(paid_media_spends)]["xDecompPerc"] / temp[temp["rn"].isin(paid_media_spends)]["xDecompPerc"].sum()
    
    dt_decompSpendDist_merge = pd.concat([dt_decompSpendDist_dt["rn"].reset_index(drop=True),dt_spendShare.filter(regex='_spend|_share', axis=1)],axis=1)

    dt_decompSpendDist = pd.merge(
        decomp_collect["xDecompAgg"][["rn","coef","xDecompAgg"]][decomp_collect["xDecompAgg"][["rn","coef","xDecompAgg"]]["rn"].isin(InputCollect["paid_media_spends"])].reset_index(drop=True),
        dt_decompSpendDist_merge.reset_index(drop=True),#dt_decompSpendDist.filter(regex='_spend|_share', axis=1),
        on="rn",
        how="left")
    dt_decompSpendDist["effect_share"] = effect_share
    
    if not refresh:
        decomp_rssd = np.sqrt(np.sum((effect_share.to_numpy().flatten().tolist() - dt_decompSpendDist["spend_share"].to_numpy())**2))
        # Penalty for models with more 0-coefficients
        if rssd_zero_penalty:
            share_0eff = sum(effect_share == 0) / len(effect_share)
            decomp_rssd = decomp_rssd * (1 + share_0eff)
        else:
            xDecompPerc_df = pd.DataFrame(decomp_collect["xDecompAgg"][["rn","xDecompPerc"]]).reset_index()
            xDecompPerc_df.columns = ["rn","xDecompPerc"]
            dt_decompRF = (
                xDecompPerc_df
                .merge(
                    decomp_collect["xDecompAggPrev"][["rn", "xDecompPerc"]],
                    on="rn",
                    how="left",
                    suffixes=("", "_prev")))
            decomp_rssd_media = dt_decompRF[dt_decompRF["rn"].isin(InputCollect["paid_media_spends"]), :]. \
                assign(diff_decomp_perc=lambda x: x["decomp_perc"] - x["decomp_perc_prev"]) \
                    .pipe(lambda x: np.sqrt(np.mean(x["diff_decomp_perc"] ** 2))) \
                        .item()

            decomp_rssd_nonmedia = dt_decompRF[~dt_decompRF["rn"].isin(InputCollect["paid_media_spends"]),].agg(rssd_nonmedia=("decomp_perc", lambda x: np.sqrt(np.mean((x - x.shift(1).fillna(0)) ** 2))))['rssd_nonmedia'][0]

            # Caution: refresh_stepsはrobyn_engineeringの機能中（robyn_args <- setdiff(~) in inputs.R）にて追加実装必要
            decomp_rssd = decomp_rssd_media + decomp_rssd_nonmedia / (1 - refresh_steps / InputCollect["rollingWindowLength"])
    
        # When all media in this iteration have 0 coefficients
        if decomp_rssd is None:
            decomp_rssd = float("inf")
            effect_share = 0

        #####################################
        #### Collect Multi-Objective Errors and Iteration Results
        resultCollect = {}
        # Auxiliary dynamic vector
        lng_tmp = i+1
        common_1 = {
            "rsq_train" : mod_output["rsq_train"],
            "rsq_val" : mod_output["rsq_val"],
            "rsq_test" : mod_output["rsq_test"],
            "nrmse_train" : mod_output["nrmse_train"],
            "nrmse_val" : mod_output["nrmse_val"],
            "nrmse_test" : mod_output["nrmse_test"],
            "nrmse" : nrmse,
            "decomp.rssd" : decomp_rssd,
            "mape" : mape,
            "_lambda" : lambda_scaled,
            "lambda_hp" : lambda_hp,
            "lambda_max" : lambda_max,
            "lambda_min_ratio" : lambda_min_ratio,
            "lg_mod" : mod_output["lg_mod"]
            }
        common_2 = {
            "solID" : f"{trial}_{lng_tmp}_{i}",
            "trial" : trial,
            "iterNG" : lng_tmp,
            "iterPar" : i}
        
        resultCollect["resultHypParam"] = dict(hypParamSam.pop("lambda"), common_1)
        resultCollect["resultHypParam"]["pos"] = decomp_collect["xDecompAgg"]["pos"].prod()
        resultCollect["resultHypParam"]["Elapsed"] = (dt.now() - t1).total_seconds()
        resultCollect["resultHypParam"]["ElapsedAccum"] = (dt.now() - t0).total_seconds()
        resultCollect["resultHypParam"] = dict(resultCollect["resultHypParam"], common_2)

        resultCollect["xDecompAgg"] = decomp_collect["xDecompAgg"]
        resultCollect["xDecompAgg"]["train_size"] = train_size
        # hypParamSam = hypParamSam.drop(columns=["lambda"])#.reset_index(drop=True)
        # common["hypParamSam"] = hypParamSam
        
        # if InputCollect["calibration_input"] is not None:
        #     resultCollect["liftCalibration"] = liftCollect.merge(common, left_index=True, right_index=True)
        
        resultCollect["decompSpendDist"] = dt_decompSpendDist
        
        return resultCollect

def robyn_mmm(InputCollect,hyper_collect,iterations,nevergrad_algo,intercept_sign,ts_validation = True
                ,add_penalty_factor = False,dt_hyper_fixed = None,rssd_zero_penalty = True
                ,refresh = False,trial = 1,seed = 123,quiet = False):
    ################################################
    #### Collect hyperparameters
    if True:
        hypParamSamName = hyper_collect["hyper_list_all"]
        # Optimization hyper-parameters
        hyper_bound_list_updated = hyper_collect["hyper_list_all"]
        hyper_bound_list_updated_name = hyper_collect["hyper_bound_list_updated"]
        hyper_count = len(hyper_bound_list_updated_name)
        # Fixed hyper-parameters
        hyper_bound_list_fixed = hyper_collect["hyper_list_all"] #hyper_collect["hyper_bound_list_fixed"]
        hyper_bound_list_fixed_name = hyper_collect["hyper_bound_list_fixed"] #hyper_bound_list_fixed
        hyper_count_fixed = len(hyper_bound_list_fixed_name)
        dt_hyper_fixed_mod = hyper_collect["dt_hyper_fixed_mod"].reset_index(drop=True)
        hyper_fixed = hyper_collect["all_fixed"]
    
    ## Get environment for parallel backend
    if True:
        # self.nevergrad_algo = nevergrad_algo
        # self.ts_validation = ts_validation
        # self.add_penalty_factor = add_penalty_factor
        # self.intercept_sign = intercept_sign
        i = None
        rssd_zero_penalty = rssd_zero_penalty
        if InputCollect["refresh_steps"] is None: refresh_steps = 0
        else: refresh_steps = InputCollect["refresh_steps"]
        # self.trial = trial

    ################################################
    #### Setup environment
    if InputCollect["dt_mod"] is None:
        raise LookupError("Run InputCollect$dt_mod = robyn_engineering() first to get the dt_mod")
    
    ################################################
    #### Get spend share
    dt_inputTrain = InputCollect["dt_input"].loc[InputCollect["rollingWindowStartWhich"]:InputCollect["rollingWindowEndWhich"]]
    temp = dt_inputTrain[InputCollect["paid_media_spends"]]
    dt_spendShare = pd.DataFrame({
        'rn': InputCollect["paid_media_spends"],
        'total_spend': temp.sum(axis=0),
        'mean_spend': temp.mean(axis=0)
        })
    dt_spendShare['spend_share'] = dt_spendShare['total_spend'] / dt_spendShare['total_spend'].sum()
    refreshAddedStartWhich = InputCollect["dt_modRollWind"].index[InputCollect["dt_modRollWind"]['ds']==InputCollect["refreshAddedStart"]].tolist()[0]
    
    temp = dt_inputTrain[InputCollect["paid_media_spends"]].loc[refreshAddedStartWhich:refreshAddedStartWhich+InputCollect["rollingWindowLength"]]

    dt_spendShareRF = pd.DataFrame({
        'rn': InputCollect["paid_media_spends"],
        'total_spend': temp.sum(axis=0),
        'mean_spend': temp.mean(axis=0)
        })

    dt_spendShareRF['spend_share'] = dt_spendShareRF['total_spend'] / dt_spendShareRF['total_spend'].sum()
    dt_spendShare = dt_spendShare.merge(dt_spendShareRF, on='rn', suffixes=('', '_refresh'), how='left')

    ################################################
    #### Get lambda
    lambda_min_ratio = 0.0001 # default  value from glmnet
    lambdas = lambda_seq(x = InputCollect["dt_mod"].drop(["ds","dep_var"],axis=1),
                         y = InputCollect["dt_mod"]["dep_var"],seq_len = 100,
                         lambda_min_ratio = lambda_min_ratio)

    lambda_max = max(lambdas) * 0.1
    lambda_min = lambda_max * lambda_min_ratio

    ################################################
    #### Start Nevergrad loop
    t0 = dt.now()

    ## Set iterations
    if hyper_fixed == False: iterTotal = iterations
    else: iterTotal = iterPar = iterNG = 1
    ## Start Nevergrad optimizer
    if not hyper_fixed:
        my_tuple = [hyper_count]#(hyper_count,)
        instrumentation = ng.p.Array(shape = my_tuple, lower = 0, upper = 1)
        # optimizer = ng.optimizers.registry[nevergrad_algo](instrumentation, budget = iterTotal, num_workers = cores)
        optimizer = ng.optimizers.registry[nevergrad_algo](instrumentation, budget = iterTotal)
        # Set multi-objective dimensions for objective functions (errors)
        # if calibration_input is None:
        optimizer.tell(ng.p.MultiobjectiveReference(), [1, 1])
        # else:
        #     optimizer.tell(ng.p.MultiobjectiveReference(), [1, 1, 1])
    ## Prepare loop
    resultCollect = {}
    resultCollect["resultHypParam"] = []
    resultCollect["xDecompAgg"] = []
    resultCollect["decompSpendDist"] = []

    # nrmse_collect = []
    # mape_lift_collect = []
    # decomp_rssd_collect = []

    for lng in tqdm(range(1,iterTotal+1),total=iterTotal, desc='Progress', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',disable=quiet):
        # self.lng = lng+1
        nevergrad_hp = {}
        nevergrad_hp_val = {}
        hypParamSamList = pd.DataFrame()
        hypParamSamNG = {}
        
        if hyper_fixed == False:
            nevergrad_hp[lng] = optimizer.ask()
            nevergrad_hp_val[lng] = [s for s in nevergrad_hp[lng].value]
            ## Scale sample to given bounds using uniform distribution
            for hypNameLoop in hyper_bound_list_updated_name:
                index = hyper_bound_list_updated_name.index(hypNameLoop) #np.where(hypNameLoop == hyper_bound_list_updated_name)
                
                channelBound = hyper_bound_list_updated[hypNameLoop]#.tolist()[0]
                hyppar_value = nevergrad_hp_val[lng]
                    
                hyppar_value = round(nevergrad_hp_val[lng][index], 6)
                if len(channelBound) > 1: hypParamSamNG[hypNameLoop] = uniform.ppf(hyppar_value, min(channelBound), max(channelBound))
                else: hypParamSamNG[hypNameLoop] = hyppar_value
            
            each_hypParamSamList = pd.DataFrame(data=hypParamSamNG.values(),index=hypParamSamNG.keys()).T
            # each_hypParamSamList["iter"] = co 
            each_hypParamSamList["iter"] = lng 
            hypParamSamList = pd.concat([hypParamSamList,each_hypParamSamList])

            hypParamSamNG = hypParamSamList.copy().reset_index(drop=True)

            ## Add fixed hyperparameters
            if hyper_count_fixed != 0:
                hypParamSamName_col = [s for s in hypParamSamName.keys()]
                hypParamSamNG = pd.concat([hypParamSamNG,dt_hyper_fixed_mod],axis=1)
                hypParamSamNG = hypParamSamNG.loc[:, hypParamSamName_col]
            else:
                hypParamSamNG = dt_hyper_fixed_mod[hypParamSamName_col]

        ########### start
        # if calibration_input is not None: lift_cal_collect = {}

        doparCollect = robyn_iterations(lng,hypParamSamNG,lambda_min,lambda_max,lambda_min_ratio,trial,add_penalty_factor,causal_mod,ts_validation)
        # nrmse_collect.append(doparCollect["resultHypParam"]["nrmse"])
        # decomp_rssd_collect.append(doparCollect["resultHypParam"]["decomp.rssd"])
        # mape_lift_collect.append(doparCollect["resultHypParam"]["mape"])
        #####################################
        #### Nevergrad tells objectives
        if not hyper_fixed:
            # if calibration_input is None:
            # optimizer.tell(nevergrad_hp[lng], [nrmse_collect[lng-1], decomp_rssd_collect[lng-1]])
            optimizer.tell(nevergrad_hp[lng], [doparCollect["resultHypParam"]["nrmse"], doparCollect["resultHypParam"]["decomp.rssd"])])
            # else:
            #     optimizer.tell(nevergrad_hp[lng], [nrmse_collect[lng-1], decomp_rssd_collect[lng-1], mape_lift_collect[lng-1]])
        
        resultCollect["resultHypParam"].append(doparCollect["resultHypParam"])
        resultCollect["xDecompAgg"].append(doparCollect["xDecompAgg"]) #xDecompAggCollect
        resultCollect["decompSpendDist"].append(doparCollect["decompSpendDist"]) #decompSpendCollect
        
        # if calibration_input is not None: resultCollectNG[lng]["liftCalibration"] = lift_cal_collect


    return resultCollect

def robyn_train(
        InputCollect,
        hyper_collect,
        iterations,
        trials,
        intercept_sign,
        nevergrad_algo,
        dt_hyper_fixed = None,
        ts_validation = True,
        add_penalty_factor = False,
        rssd_zero_penalty = True,
        refresh = False,
        seed = 123,
        quiet = False
):
    hyper_fixed = hyper_collect["all_fixed"]

    if hyper_fixed:
        OutputModels = {}
        OutputModels["trial1"] = robyn_mmm(InputCollect, hyper_collect = hyper_collect,iterations = iterations,nevergrad_algo = nevergrad_algo,
                                 intercept_sign = intercept_sign,dt_hyper_fixed = dt_hyper_fixed,ts_validation = ts_validation,
                                 add_penalty_factor = add_penalty_factor,rssd_zero_penalty = rssd_zero_penalty,seed = seed,
                                 quiet = quiet)
        OutputModels["trial"] = 1

        if "solID" in dt_hyper_fixed:
            these = list("resultHypParam", "xDecompVec", "xDecompAgg", "decompSpendDist")
            for tab in these:
                OutputModels[tab]["solID"] = dt_hyper_fixed["solID"]
    else:
        # ck.check_init_msg(cores)
        if not quiet: print(f">>> Starting {trials} trials with {iterations} iterations each using with calibration using {nevergrad_algo} nevergrad algorithm...")
    
        OutputModels = {}
        
        for ngt in range(1,trials+1):
            if not quiet:
                print(f"  Running trial {ngt} of {trials}")
            model_output = robyn_mmm(InputCollect, hyper_collect = hyper_collect,iterations = iterations,nevergrad_algo = nevergrad_algo,
                                        intercept_sign = intercept_sign,ts_validation = ts_validation,add_penalty_factor = add_penalty_factor,
                                        rssd_zero_penalty = rssd_zero_penalty,refresh = refresh,trial = ngt,seed = seed + ngt,
                                        quiet = quiet,causal_mod = causal_mod)

            check_coef0 = any([s == float('inf') for s in model_output["resultHypParam"]["decomp_rssd"]])
            if check_coef0:
                num_coef0_mod = model_output["decompSpendDist"].loc[model_output["resultHypParam"]["decomp.rssd"].isin([np.inf, -np.inf])].drop_duplicates(["iterNG", "iterPar"]).shape[0]
                num_coef0_mod = iterations if num_coef0_mod > iterations else num_coef0_mod
            model_output["trial"] = ngt
            OutputModels[f"trial{ngt}"] = model_output
    return OutputModels

def robyn_run(
       InputCollect,
       dt_hyper_fixed = None,
       ts_validation = False,
       add_penalty_factor = False,
       refresh = False,
       seed = 123,
       outputs = False,
       quiet = False,
       cores = None,
       trials = 5,
       iterations = 2000,
       rssd_zero_penalty = True,
       nevergrad_algo = "TwoPointsDE",
       intercept_sign = "non_negative",
       lambda_control = None 
):

    t0 = dt.now()
    # Use previously exported model (Consider to add in the future)

    if InputCollect["hyper_params"] is None: raise ValueError("Must provide 'hyperparameters' in robyn_inputs()'s output first")
    
    hyps_fixed = dt_hyper_fixed is not None

    if hyps_fixed: trials = iterations = 1
    
    ck.check_run_inputs(InputCollect, cores,iterations,trials,intercept_sign,nevergrad_algo)

    # currently unable to calibrate
    calibration_input = None
    
    # ck.check_iteration(iterations,calibration_input,trials,hyps_fixed,refresh)
    #ref.init_msgs_run(refresh,lambda_control, quiet)

    #####################################
    #### Prepare hyper-parameters
    hyper_collect = hp.hyper_collector(InputCollect["adstock"],InputCollect["all_media"],InputCollect["dt_mod"]
                                        ,InputCollect["hyper_params"],ts_validation,add_penalty_factor,dt_hyper_fixed,cores)

    hyper_updated = hyper_collect["hyper_list_all"]

    robyn_outputs = robyn_train(
        InputCollect,
        hyper_collect,
        iterations = iterations,
        trials = trials,
        intercept_sign = intercept_sign,
        nevergrad_algo = nevergrad_algo,
        dt_hyper_fixed = dt_hyper_fixed,
        ts_validation = ts_validation,
        add_penalty_factor = add_penalty_factor,
        rssd_zero_penalty = rssd_zero_penalty,
        refresh = refresh,
        seed = seed,
        quiet = quiet
    )

    robyn_outputs["hyper_fixed"] = hyper_collect["all_fixed"]
    # self.OutputModels.attr["bootstrap"] = bootstrap
    robyn_outputs["refresh"] = refresh

    if True:
        robyn_outputs["cores"] = cores
        robyn_outputs["iterations"] = iterations
        robyn_outputs["trials"] = trials
        robyn_outputs["intercept_sign"] = intercept_sign
        robyn_outputs["nevergrad_algo"] = nevergrad_algo
        robyn_outputs["ts_validation"] = ts_validation
        robyn_outputs["add_penalty_factor"] = add_penalty_factor
        robyn_outputs["hyper_updated"] = hyper_collect["hyper_list_all"]
            
    # Not direct output & not all fixed hyperparameters
    if not outputs and dt_hyper_fixed is None: output = robyn_outputs
    elif not hyper_collect["all_fixed"]: raise NotImplementedError("The process is under implementing...")
    else: raise NotImplementedError("The process is under implementing...")
    
    # Check convergence when more than 1 iteration
    if not hyper_collect["all_fixed"]:
        # raise NotImplementedError
        print("The process is under implementing...")
        pass
    else:
        if "solID" in dt_hyper_fixed: output["solID"] = dt_hyper_fixed["solID"]
        else: output["selectID"] = robyn_outputs["trial1"]["resultHypParam"]["solID"]
        if not quiet: print(f"Successfully recreated model ID: {output.selectID}")
    
    # Save hyper-parameters list
    output["hyper_updated"] = hyper_collect["hyper_list_all"]
    output["seed"] = seed
    
    return output

def model_refit(self,x_train,y_train,x_val,y_val,x_test,y_test,lambda_,lower_limits,upper_limits,intercept_sign = "non_negative",penalty_factor = None):
    if penalty_factor is None: penalty_factor = np.ones(y_train.shape[1])
    
    # mod = glmnet(x_train, y_train, alpha=0, lambdau=lambda_, lower_limits = lower_limits, upper_limits=upper_limits, scoring = "mean_squared_error", penalty_factor=penalty_factor)
    mod = Ridge(alpha=0)
    mod.fit(x_train, y_train)
    
    df_int = 1

    if intercept_sign == "non_negative" and mod.coef_[0]<0:
        # mod = glmnet(x_train, y_train, alpha = 0, lambdau = lambda_, lower_limits = lower_limits, upper_limits = upper_limits, penalty_factor = penalty_factor, fit_intercept = False)
        mod = Ridge(alpha=0,fit_intercept=False)
        mod.fit(x_train, y_train)

        df_int = 1 

    # Calculate all Adjusted R2
    y_train_pred = mod.predict(x_train)
    rsq_train = lm.get_rsq_py(true = y_train, predicted = y_train_pred, p = x_train.shape[1], df_int = df_int)
    if x_val is not None:
        y_val_pred = mod.predict(x_val) 
        rsq_val = lm.get_rsq_py(true = y_val, predicted = y_val_pred, p = x_val.shape[1], df_int = df_int, n_train = len(y_train))
        y_test_pred = mod.predict(x_test)
        rsq_test = lm.get_rsq_py(true = y_test, predicted = y_test_pred, p = x_test.shape[1], df_int = df_int, n_train = len(y_train))
        y_pred = np.concatenate((y_train_pred, y_val_pred, y_test_pred))
    else:
        rsq_val = rsq_test = None
        y_pred = y_train_pred
    
    # Calculate all NRMSE
    nrmse_train = np.sqrt(np.mean((y_train - y_train_pred)**2)) / (max(y_train) - min(y_train))
    if x_val is not None:
        nrmse_val = np.sqrt(np.mean(sum((y_val - y_val_pred)**2))) / (max(y_val) - min(y_val))
        nrmse_test = np.sqrt(np.mean(sum((y_test - y_test_pred)**2))) / (max(y_test) - min(y_test))
    else:
        nrmse_val = nrmse_test = y_val_pred = y_test_pred = None
    
    coef_df = pd.DataFrame(mod.coef_).T
    coef_df.columns = mod.feature_names_in_

    return {"rsq_train" : rsq_train,"rsq_val" : rsq_val,"rsq_test" : rsq_test,"nrmse_train" : nrmse_train
            ,"nrmse_val" : nrmse_val,"nrmse_test" : nrmse_test,"coefs" : mod.coef_,"mod" : mod
            ,"y_train_pred" : y_train_pred,"y_val_pred" : y_val_pred,"y_test_pred" : y_test_pred
            ,"y_pred" : y_pred,"df_int" : df_int,"coef_df" : coef_df}


