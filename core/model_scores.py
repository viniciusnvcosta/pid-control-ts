"""
Generates forecasts from an ARIMA model
"""

import contextlib
import copy
import os
import warnings

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models.forecasting.arima import ARIMA
from darts.models.forecasting.prophet_model import Prophet
from darts.models.forecasting.theta import Theta
from darts.models.forecasting.transformer_model import TransformerModel


def generate_forecasts(
    data, model_name, savename, overwrite, log, fit_every, ahead, *args, **kwargs
):
    if not overwrite:
        # A missing or unreadable cache means the forecasts are regenerated.
        with contextlib.suppress(OSError, KeyError, ValueError):
            saved = np.load(savename)
            forecasts = saved["forecasts"]
            return forecasts
    T = data.shape[0]
    forecasts = np.zeros((T,))
    data2 = copy.deepcopy(data)
    if log:
        data2["y"] = np.log(data2["y"])
    # Create a new timestamp that is daily
    data2.index = pd.date_range(start=data2.index.min(), periods=len(data2), freq="D")
    y = TimeSeries.from_dataframe(
        data2["y"].interpolate().reset_index(drop=False).sort_values(by="index"),
        time_col="index",
        value_cols="y",
    )
    # Generate the forecasts
    print("Generating forecasts...")
    model = None
    if model_name == "prophet":
        model = Prophet()
    elif model_name == "ar":
        model = ARIMA(p=3, d=0, q=0)
    elif model_name == "theta":
        if fit_every > 1:
            raise ValueError("Theta does not support fit_every > 1")
        model = Theta()
    elif model_name == "transformer":
        model = TransformerModel(12, ahead, n_epochs=10)
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        # WARNING: This doesn't always work. If not, make sure to execute on your system to use the transformer architecture.
        y = y.astype("float32")
    else:
        raise ValueError("Invalid model name")
    # Ignore ConvergenceWarning
    warnings.filterwarnings("ignore", category=UserWarning)
    retrain = fit_every if model_name != "transformer" else fit_every * 100
    model_forecasts = (
        model.historical_forecasts(
            y, forecast_horizon=fit_every, retrain=retrain, verbose=True
        )
        .values()[:, -1]
        .squeeze()
    )
    forecasts[-model_forecasts.shape[0] :] = model_forecasts
    if log:
        forecasts = np.exp(forecasts)
    print("Finished generating forecasts.")

    # Save and return
    np.savez(savename, forecasts=forecasts)
    return forecasts
