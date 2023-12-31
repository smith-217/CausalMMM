import pandas as pd
import numpy as np
from datetime import datetime as dt
import warnings
import CausalMMM.hyper_params as hp

OPTS_PDN = ["positive", "negative", "default"]
HYPS_OTHERS = ["lambda", "train_size"]

def check_datevar(
    InputCollect,
    date_var = "auto"
):
    if date_var[0] == "auto":
        is_date_pre = InputCollect["dt_input"].applymap(lambda x: isinstance(x, pd.Timestamp)).any()
        is_date = [i for i, x in enumerate(is_date_pre) if x]
        # is_date = np.where(np.array(list(map(lambda x: isinstance(x, np.datetime64), dt_input))).flatten())[0]
        if len(is_date) == 1:
            date_var = list(is_date)
            print(f"Automatically detected 'date_var':{date_var}")
        else:
            raise ValueError("Can't automatically find a single date variable to set 'date_var'")
    if date_var is None or len(date_var) > 1 or date_var not in list(self.dt_input):
        raise ValueError("You must provide only 1 correct date variable name for 'date_var'")
    
    dt_input_sorted = InputCollect["dt_input"].sort_values(by=date_var)
    date_var_dates = list(
        dt_input_sorted[date_var][0]
        ,dt_input_sorted[date_var][1])
    
    if any(date_var_dates.value_counts()) > 1:
        raise ValueError("Date variable shouldn't have duplicated dates (panel data)")
    if any(date_var_dates is None) or any(np.isinf(date_var_dates)):
        raise ValueError("Dates in 'date_var' must have format '2020-12-31' and can't contain NA nor Inf values")
        
    dayInterval = dt.strptime(date_var_dates[1], '%Y-%m-%d').date() - dt.strptime(date_var_dates[0], '%Y-%m-%d').date()
    dayInterval = int(dayInterval)
    
    if dayInterval == 1:
        intervalType = "day"
    elif dayInterval == 7:
        intervalType = "week"
    elif dayInterval in range(28,31):
        intervalType = "month"
    else:
        raise ValueError(f"{date_var} data has to be daily, weekly or monthly")
    
    InputCollect["dt_input"] = dt_input_sorted
    InputCollect["date_var"] = date_var
    InputCollect["dayInterval"] = dayInterval
    InputCollect["intervalType"] = intervalType

    return InputCollect

def check_prophet(
        InputCollect,
        prophet_country,
        prophet_signs
):
    # check_vector(prophet_vars)
    # check_vector(prophet_signs)
    if InputCollect["dt_holidays"] is None or InputCollect["prophet_vars"] is None:
        return np.nan
    else:
        InputCollect["prophet_vars"] = InputCollect["prophet_vars"].lower()
        opts = list("trend", "season", "monthly", "weekday", "holiday")
    if not all(item in opts for item in InputCollect["prophet_vars"]):
      raise ValueError(f"Allowed values for 'prophet_vars' are: {opts}")
    if "weekday" in InputCollect["prophet_vars"] and InputCollect["dayInterval"] > 7:
        warnings.warn("Ignoring prophet_vars = 'weekday' input given your data granularity")
    if prophet_country is None or len(prophet_country) > 1 or prophet_country not in InputCollect["dt_holidays"]["country"].unique():
      raise ValueError(
        f"You must provide 1 country code in 'prophet_country' input. {len(InputCollect['dt_holidays']['country'].unique())} countries are included: {InputCollect['dt_holidays']['country'].unique()} \n If your country is not available, please manually add it to 'dt_holidays'"
        )    
    if prophet_signs is None:
        prophet_signs = ["default"] * len(InputCollect["prophet_vars"])
    if not all(prophet_signs in OPTS_PDN):
        raise ValueError(f"Allowed values for 'prophet_signs' are: {OPTS_PDN}")
    if len(prophet_signs) != len(InputCollect["prophet_vars"]):
        raise ValueError("'prophet_signs' must have same length as 'prophet_vars'")

    InputCollect["prophet_signs"] = prophet_signs
    return InputCollect

def check_context(
        InputCollect,
        context_signs
):
    if InputCollect["context_vars"] is None:
        if context_signs is None:
            context_signs = ["default"] * len(InputCollect["context_vars"])
        if not all(context_signs in OPTS_PDN):
            raise ValueError(f"Allowed values for 'context_signs' are: {OPTS_PDN}")
        if len(context_signs) != len(InputCollect["context_vars"]):
            raise ValueError("Input 'context_signs' must have same length as 'context_vars'")
        
        temp = InputCollect["context_vars"] in list(InputCollect["dt_input"])
        if not all(temp):
            raise ValueError("Input 'context_vars' not included in data.")
        InputCollect["context_signs"] = context_signs
    
    return InputCollect

def check_paidmedia(InputCollect,paid_media_signs):
    if InputCollect["paid_media_spends"] is None:
        raise AttributeError("Must provide 'paid_media_spends'")
    # check_vector(paid_media_vars)
    # check_vector(paid_media_signs)
    # check_vector(paid_media_spends)
    mediaVarCount = len(InputCollect["paid_media_vars"])
    spendVarCount = len(InputCollect["paid_media_spends"])
    temp = InputCollect["paid_media_vars"] in list(InputCollect["dt_input"])
    if not all(temp):
        raise LookupError("Input 'paid_media_vars' not included in data.")
    temp = InputCollect["paid_media_spends"] in list(InputCollect["dt_input"])
    if not all(temp):
        raise LookupError("Input 'paid_media_spends' not included in data.")
    if paid_media_signs is None:
        paid_media_signs = ["positive"] * len(mediaVarCount)
    if not all(paid_media_signs in OPTS_PDN):
        raise ValueError(f"Allowed values for 'paid_media_signs' are: {OPTS_PDN}")
    if len(paid_media_signs) == 1:
        paid_media_signs = paid_media_signs * len(InputCollect["paid_media_vars"])
    if len(paid_media_signs) != len(InputCollect["paid_media_vars"]):
        raise IndexError("Input 'paid_media_signs' must have same length as 'paid_media_vars'")
    if spendVarCount != mediaVarCount:
        raise IndexError("Input 'paid_media_spends' must have same length as 'paid_media_vars'")
    is_num = np.array(InputCollect["dt_input"][InputCollect["paid_media_vars"]].applymap(np.isnumeric)).flatten()
    if not all(is_num):
        raise TypeError("All your 'paid_media_vars' must be numeric.")
    all_paid_media_vars = list(set(InputCollect["paid_media_vars"]) | set(InputCollect["paid_media_spends"]))
    get_cols = any(InputCollect["dt_input"][all_paid_media_vars].values < 0)
    if get_cols:
        check_media_names = np.unique(InputCollect["paid_media_vars"] + InputCollect["paid_media_spends"])
        df_check = InputCollect["dt_input"][check_media_names]
        check_media_val = df_check.apply(lambda x: any(x < 0)).tolist()
        raise ValueError("contains negative values. Media must be >=0")
    
    InputCollect["paid_media_signs"] = paid_media_signs
    InputCollect["mediaVarCount"] = mediaVarCount
    return InputCollect

def check_organicvars(
    InputCollect
    ,organic_signs
):
    if InputCollect["organic_vars"] is None:
        return np.nan
    # check_vector(organic_vars)
    # check_vector(organic_signs)
    temp = InputCollect["organic_vars"] in list(InputCollect["dt_input"])
    if not all(temp):
        raise ValueError("Input 'organic_vars' not included in data.")
    if InputCollect["organic_vars"] is not None & organic_signs is None:
        organic_signs = ["positive"] * len(InputCollect["organic_vars"])
    if not all(organic_signs in OPTS_PDN):
        raise ValueError(f"Allowed values for 'organic_signs' are: {OPTS_PDN}")
    if len(organic_signs) != len(InputCollect["organic_vars"]):
        raise ValueError("Input 'organic_signs' must have same length as 'organic_vars'")
    
    InputCollect["organic_signs"] = organic_signs
    return InputCollect

def check_allvars(all_ind_vars):
    if len(all_ind_vars) != len(np.unique(all_ind_vars)):
        raise IndexError("All input variables must have unique names")

def check_windows(InputCollect,window_start,window_end):
    dates_vec = pd.to_datetime(InputCollect["dt_input"][InputCollect["date_var"]]).dt.strftime('%Y-%m-%d')
    
    if window_start is None:
        window_start = dates_vec.min()
    else:
        window_start = dt.strptime(window_start, '%Y-%m-%d').date()
        if window_start is None:
            raise TypeError("Input 'window_start' must have date format.")
        elif window_start < dates_vec.min():
            window_start = dates_vec.min()
            warnings.warn(f"Input 'window_start' is smaller than the earliest date in input data. It's automatically set to the earliest date:{window_start}")
        elif window_start > dates_vec.max():
            raise ValueError(f"Input 'window_start' can't be larger than the the latest date in input data: {dates_vec.max()}")
    
    rollingWindowStartWhich = np.argmin(np.abs(dates_vec - window_start)).tolist()
    if window_start not in dates_vec:
        window_start = InputCollect["dt_input"].loc[rollingWindowStartWhich, InputCollect["date_var"]]
        warnings.warn(f"Input 'window_start' is adapted to the closest date contained in input data: {window_start}")
    refreshAddedStart = window_start
    
    if window_end is None:
        window_end = dates_vec.max()
    else:
        window_end <- dt.strptime(window_end, '%Y-%m-%d').date()
        if window_end is None:
            raise ValueError("Input 'window_end' must have date format,.")
        elif window_end > dates_vec.max():
            window_end = dates_vec.max()
            warnings.warn(f"Input 'window_end' is larger than the latest date in input data. It's automatically set to the latest date: {window_end}" )
        elif window_end < window_start:
            window_end = dates_vec.max()
            warnings.warn(f"Input 'window_end' must be >= 'window_start. It's automatically set to the latest date: {window_end}")
    
    rollingWindowEndWhich = np.argmin(np.abs(dates_vec - window_end))
    if window_end not in dates_vec:
        window_end = InputCollect["dt_input"].loc[rollingWindowEndWhich, InputCollect["date_var"]]
        warnings.warn(f"Input 'window_end' is adapted to the closest date contained in input data: {window_end}")
    rollingWindowLength = rollingWindowEndWhich - rollingWindowStartWhich + 1
    
    dt_init = InputCollect["dt_input"].loc[rollingWindowStartWhich:rollingWindowEndWhich, InputCollect["all_media"]]
    
    dt_init_numeric = dt_init.select_dtypes(include=np.number)
    init_all0 = (dt_init_numeric.sum() == 0).all()
    if any(init_all0):
        raise ValueError("These media channels contains only 0 within training period ")
    
    InputCollect["window_start"] = window_start
    InputCollect["rollingWindowStartWhich"] = rollingWindowStartWhich
    InputCollect["refreshAddedStart"] = refreshAddedStart
    InputCollect["window_end"] = window_end
    InputCollect["rollingWindowEndWhich"] = rollingWindowEndWhich
    InputCollect["rollingWindowLength"] = rollingWindowLength
    return InputCollect

def check_run_inputs(
        iterations = None,
        trials = None,
        intercept_sign = None,
        nevergrad_algo = None
):
    if iterations is None:
        raise ValueError("Must provide 'iterations' in robyn_run()")
    if trials is None:
        raise ValueError("Must provide 'trials' in robyn_run()")
    if nevergrad_algo is None:
        raise ValueError("Must provide 'nevergrad_algo' in robyn_run()")
    opts = ["non_negative", "unconstrained"]
    if intercept_sign not in opts:
        raise ValueError(f"Input 'intercept_sign' must be any of: {opts}")

def check_iteration(
        InputCollect,
        iterations = None,
        trials = None,
        hyps_fixed = False,
        refresh = False
):
    if not refresh:
        if not hyps_fixed:
            if InputCollect["calibration_input"] is None & iterations < 2000 or trials < 5:
                warnings.warn("We recommend to run at least 2000 iterations per trial and 5 trials to build initial model")
            elif InputCollect["calibration_input"] is not None & iterations < 2000 or trials < 10:
                warnings.warn("You are calibrating MMM. We recommend to run at least 2000 iterations per trial and 10 trials to build initial model")

def check_hyper_fixed(
        InputCollect,
        dt_hyper_fixed,
        add_penalty_factor
):
    hyper_fixed = {}
    hyper_fixed["hyper_fixed"] = dt_hyper_fixed is not None
    
    # Adstock hyper-parameters
    hypParamSamName = hp.hyper_names(InputCollect)
    
    # Add lambda and other hyper-parameters manually
    hypParamSamName = [hypParamSamName, HYPS_OTHERS]
    
    # Add penalty factor hyper-parameters names
    if add_penalty_factor:
        for_penalty = InputCollect["dt_mod"].drop(["ds","dep_var"],axis=1).columns
        hypParamSamName = [hypParamSamName, f"{for_penalty}_penalty"]
    
    if hyper_fixed:
        ## Run robyn_mmm if using old model result tables
        if dt_hyper_fixed is not None:
            if dt_hyper_fixed.shape[0] != 1:
                raise ValueError("Provide only 1 model / 1 row from OutputCollect$resultHypParam or pareto_hyperparameters.csv from previous runs")
            if not all(hypParamSamName in dt_hyper_fixed.columns):
                remove_col = hypParamSamName in dt_hyper_fixed.columns
                these = hypParamSamName.drop(remove_col,axis=1)
                raise ValueError("Input 'dt_hyper_fixed' is invalid.")
    
    hyper_fixed["hypParamSamName"] = hypParamSamName
    
    return hyper_fixed
