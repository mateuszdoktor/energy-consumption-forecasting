import os

import joblib
import numpy as np
import pmdarima as pm

HORIZONS = [24, 168]


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    errors = y_true - y_pred
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape": float(np.mean(np.abs(errors / y_true)) * 100),
    }


def naive_single_origin(y_history, horizon, lag):
    pattern = y_history.iloc[-lag:].values
    return np.array([pattern[(k - 1) % lag] for k in range(1, horizon + 1)])


def _metrics_row(model, split_name, horizon, y_true, y_pred):
    return {
        "model": model,
        "split": split_name,
        "horizon": horizon,
        **regression_metrics(y_true, y_pred),
    }


def run_evaluation(model, y_train, splits, horizons=HORIZONS):
    rows = []
    for split_name, (X_split, y_split) in splits.items():
        for horizon in horizons:
            h = min(horizon, len(y_split))
            lag = h
            actual = y_split.iloc[:h].values

            pred = model.predict(n_periods=h, X=X_split.iloc[:h])
            rows.append(_metrics_row("ARIMAX DHR", split_name, h, actual, pred))

            naive_pred = naive_single_origin(y_train, h, lag=lag)
            rows.append(
                _metrics_row(
                    f"Naive seasonal (lag {lag})",
                    split_name,
                    h,
                    actual,
                    naive_pred,
                )
            )
    return rows


def arimax_dhr_search_or_load(
    y, X, force_retrain=False, model_path="../models/arimax_dhr.joblib"
):
    if os.path.exists(model_path) and not force_retrain:
        print(f"[CACHE] File {model_path} already exists. Loading results...")
        return joblib.load(model_path)

    print("[COMPUTE] Searching best ARIMAX DHR order on train set...")
    results = pm.auto_arima(
        y=y,
        X=X,
        d=1,
        start_p=1,
        start_q=1,
        max_p=3,
        max_q=3,
        seasonal=False,
        information_criterion="aic",
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(results, model_path)
    print(f"[SAVE] Results saved to: {model_path}")
    return results
