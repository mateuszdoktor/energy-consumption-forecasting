from statsmodels.tsa.seasonal import MSTL
import pandas as pd
import os


def mstl_fit_or_load(series, periods, split_name):
    periods_str = "_".join(str(p) for p in periods)
    file_path = f"../models/mstl_{split_name}_{periods_str}.parquet"

    if os.path.exists(file_path):
        print(f"[CACHE] File {file_path} already exists. Loading results..")
        return pd.read_parquet(file_path)

    print(f"[COMPUTE] No results for {split_name} ({periods_str}). Running MSTL..")
    model = MSTL(endog=series, periods=periods).fit()
    mstl_results = pd.DataFrame(
        {"observed": model.observed, "trend": model.trend, "resid": model.resid}
    )
    for season in model.seasonal.columns:
        mstl_results[season] = model.seasonal[season]

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    mstl_results.to_parquet(file_path)
    print(f"[SAVE] Results saved: {file_path}")
    return mstl_results
