import pandas as pd
import numpy as np
import itertools
import CausalMMM.checks as ck
import warnings

HYPS_NAMES = ["thetas", "shapes", "scales", "alphas", "gammas"]
HYPS_OTHERS = ["lambda", "train_size"]

def hyper_names(InputCollect):
    # adstock = check_adstock(adstock)
    local_name = []
    if InputCollect["adstock"][0] == "geometric":
        for media, hyp in itertools.product(InputCollect["all_media"], HYPS_NAMES):
            if any(s in hyp for s in ['thetas', 'alphas', 'gammas']):
                local_name.append(f'{media}_{hyp}')
    elif InputCollect["adstock"][0] in ["weibull_cdf", "weibull_pdf"]:
        local_name = sorted(['_'.join(x) for x in itertools.product(InputCollect["all_media"], [x for x in HYPS_NAMES if 'shapes' in x or 'scales' in x or 'alphas' in x or 'gammas' in x])])
    
    InputCollect["local_name"] = local_name
    return local_name

def check_hyperparameters(
    InputCollect,
    hyperparameters = None,
    exposure_vars = None):
    if hyperparameters is None:
        warnings.warn("Input 'hyperparameters' not provided yet. Generate 'hyperparameters' object as None")
        InputCollect["hyperparameters"] = None
    else:
        # Non-adstock hyperparameters check
        # check_train_size(hyperparameters)
        # Adstock hyperparameters check
        hyperparameters_ordered = dict(sorted(InputCollect["hyperparameters"].items()))
        get_hyp_names = hyperparameters_ordered.keys()
        original_order = [map(lambda x: list(get_hyp_names).index(x) + 1, hyperparameters.keys())]
        ref_hyp_name_spend = hyper_names(all_media = InputCollect["paid_media_spends"])
        ref_hyp_name_expo = hyper_names(all_media = exposure_vars)
        ref_hyp_name_org = hyper_names(all_media = InputCollect["organic_vars"])
        ref_hyp_name_other = [x for x in get_hyp_names if x in HYPS_OTHERS]
        # Excluding lambda (first HYPS_OTHERS) given its range is not customizable
        # ref_all_media = sorted(ref_hyp_name_spend + ref_hyp_name_org + HYPS_OTHERS)
        all_ref_names = [ref_hyp_name_spend, ref_hyp_name_expo, ref_hyp_name_org, HYPS_OTHERS]
        all_ref_names = sorted(all_ref_names)
        if not all(get_hyp_names in all_ref_names):
            wrong_hyp_names = get_hyp_names[~np.isin(get_hyp_names, all_ref_names)]
            raise ValueError(f"Input 'hyperparameters' contains wrong names {wrong_hyp_names}.")
        total = len(get_hyp_names)
        total_in = len(list(ref_hyp_name_spend, ref_hyp_name_org, ref_hyp_name_other))
        if total != total_in:
            raise ValueError("Use hyper_names() function to help you with the correct hyperparameters names.")

def hyper_collector(
    InputCollect,
    ts_validation = False,
    add_penalty_factor = False,
    dt_hyper_fixed = None,
    cores = None
    ):
    
    hyper_in = InputCollect["hyperparameters"]

    # Fetch hyper-parameters based on media
    hypParamSamName = hyper_names(InputCollect)
    
    # Manually add other hyper-parameters
    hypParamSamName = hypParamSamName+HYPS_OTHERS
    
    # Add penalty factor hyper-parameters names
    for_penalty = pd.DataFrame(InputCollect["dt_mod"]).drop(["ds","dep_var"],axis=1).columns
    if add_penalty_factor:
        hypParamSamName = hypParamSamName + [f"penalty_{for_penalty}"]
    
    # Check hyper_fixed condition + add lambda + penalty factor hyper-parameters names
    all_fixed = ck.check_hyper_fixed(InputCollect, dt_hyper_fixed, add_penalty_factor)
    
    if not all_fixed["hyper_fixed"]:
        # Collect media hyperparameters
        hyper_bound_list = {}
        for i in range(len(hypParamSamName)):
            if hypParamSamName[i] in hyper_in:
                hyper_bound_list[hypParamSamName[i]] = hyper_in[hypParamSamName[i]]

        # Add unfixed lambda hyperparameter manually
        if "lambda" not in hyper_bound_list.keys():
            hyper_bound_list["lambda"] = [0, 1]

        # Add unfixed train_size hyperparameter manually
        if ts_validation:
            if "train_size" not in hyper_bound_list.keys():
                hyper_bound_list["train_size"] = [0.5, 0.8]
            # print(f"Time-series validation with train_size range of {100 * hyper_bound_list['train_size']}% of the data...")
        else:
            if "train_size" in hyper_bound_list.keys():
                warnings.warn("Provided train_size but ts_validation = FALSE. Time series validation inactive.")
            
            hyper_bound_list["train_size"] = 1
            print("Fitting time series with all available data...")

        # Add unfixed penalty.factor hyperparameters manually
        for_penalty = pd.DataFrame.from_dict(InputCollect["dt_mod"]).drop(["ds","dep_var"],axis=1).columns
        penalty_names = f"{for_penalty}_penalty"
        if add_penalty_factor:
            for penalty in penalty_names:
                if len(hyper_bound_list[penalty]) != 1:
                    hyper_bound_list[penalty] = [0, 1]

        # Get hyperparameters for Nevergrad
        # hyper_list_bind = {}
        hyper_bound_list_updated = {}
        hyper_bound_list_fixed = {}
        
        for x in hyper_bound_list.keys():
            if len(hyper_bound_list[x]) == 2:
                hyper_bound_list_updated[x] = hyper_bound_list[x]
            if len(hyper_bound_list[x]) == 1:
                hyper_bound_list_fixed[x] = hyper_bound_list[x]

        # # Get fixed hyperparameters
        # hyper_bound_list_fixed = [hyper_bound_list[x] for x in hyper_bound_list.keys() if len(hyper_bound_list[x]) == 1]

        hyper_list_bind = dict(hyper_bound_list_updated, **hyper_bound_list_fixed)
    
        hyper_list_all = {}
        for i in range(len(hypParamSamName)):
            if hypParamSamName[i] in hyper_list_bind:
                hyper_list_all[hypParamSamName[i]] = hyper_list_bind[hypParamSamName[i]]

        hyper_bound_list_fixed_rep = [item for sublist in hyper_bound_list_fixed for item in [sublist] * cores]
        s = pd.Series(hyper_bound_list_fixed_rep)
        dt_hyper_fixed_mod = pd.concat([s], axis=1)
        
    else:
        hyper_bound_list_fixed = {}
        for i in range(len(hypParamSamName)):
            hyper_bound_list_fixed[hypParamSamName[i]] = dt_hyper_fixed[hypParamSamName[i]]

        hyper_list_all = hyper_bound_list_fixed
        hyper_bound_list_updated = [i for i in hyper_bound_list_fixed if len(i) == 2]
        cores = 1
        
        dt_hyper_fixed_mod = pd.DataFrame(hyper_bound_list_fixed, columns=['lower', 'upper'])
        dt_hyper_fixed_mod.columns = [s for s in hyper_bound_list_fixed.keys()]

    return {
        "hyper_list_all": hyper_list_all
        ,"hyper_bound_list_updated":hyper_bound_list_updated
        ,"hyper_bound_list_fixed":hyper_bound_list_fixed
        ,"dt_hyper_fixed_mod":dt_hyper_fixed_mod
        ,"all_fixed":all_fixed}