import math

import numpy as np
import pandas as pd
# from pyfinance.ols import PandasRollingOLS


from statsmodels.datasets import longley


class IndicatorMixin:

    def _check_fillna(self, serie: pd.Series, value: int = 0):
        """Check if fillna flag is True.

        Args:
            serie(pandas.Series): dataset 'Close' column.
            value(int): value to fill gaps; if -1 fill values using 'backfill' mode.

        Returns:
            pandas.Series: New feature generated.
        """
        if self._fillna:
            serie_output = serie.copy(deep=False)
            serie_output = serie_output.replace([np.inf, -np.inf], np.nan)
            if isinstance(value, int) and value == -1:
                return serie_output.fillna(method='ffill').fillna(value=-1)
            else:
                return serie_output.fillna(method='ffill').fillna(value)
        else:
            return serie

    def _true_range(self, high: pd.Series, low: pd.Series, prev_close: pd.Series):
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.DataFrame(data={'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        return tr


def dropna(df):
    """Drop rows with "Nans" values
    """
    df = df.copy()
    number_cols = df.select_dtypes('number').columns.to_list()
    df[number_cols] = df[number_cols][df[number_cols] < math.exp(709)]  # big number
    df[number_cols] = df[number_cols][df[number_cols] != 0.0]
    df = df.dropna()
    return df


def sma(series, periods: int, fillna: bool = False):
    min_periods = 0 if fillna else periods
    return series.rolling(window=periods, min_periods=min_periods).mean()


def ema(series, periods, fillna=False):
    min_periods = 0 if fillna else periods
    return series.ewm(span=periods, min_periods=min_periods, adjust=False).mean()


def get_min_max(x1, x2, f='min'):
    """Find min or max value between two lists for each index
    """
    x1 = np.array(x1)
    x2 = np.array(x2)
    try:
        if f == 'min':
            return pd.Series(np.amin([x1, x2], axis=0))
        elif f == 'max':
            return pd.Series(np.amax([x1, x2], axis=0))
        else:
            raise ValueError('"f" variable value should be "min" or "max"')
    except Exception as e:
        return e


def pulse_conversion(x1: pd.Series, x2: pd.Series):
    """
    returns 1 on the first occurrence of "true" signal in Array1
    then returns 0 until Array2 is true even if there are "true" signals in Array1

    :param x1: Series 1
    :param x2: Series 2
    :return: Pulse array as pd.Series
    """
    x1 = np.array(x1)
    x2 = np.array(x2)
    result = np.full_like(x1, 0)

    last = False
    counter = 0
    for i, j in np.c_[x1, x2]:
        if i and not last:
            result[counter] = True
            last = True
        elif j and last:
            result[counter] = True
            last = False
        else:
            result[counter] = False
        counter = counter + 1

    return pd.Series(result)


# def linear_regression(y: pd.Series, x: pd.Series = None, window: int = 13, has_const: bool = False,
#                       use_const: bool = True, offset: bool = False):
#     """
#     Smooths out a data series using the least squares method
#
#     :param y : Y Series
#     :param x : X Series
#     :param window : int
#         Length of each rolling window
#     :param has_const : bool, default False
#         Specifies whether `x` includes a user-supplied constant (a column
#         vector).  If False, it is added at instantiation
#     :param use_const : bool, default True
#         Whether to include an intercept term in the model output.  Note the
#         difference between has_const and use_const.  The former specifies
#         whether a column vector of 1s is included in the input; the latter
#         specifies whether the model itself should include a constant
#         (intercept) term.  Exogenous data that is ~N(0,1) would have a
#         constant equal to zero; specify use_const=False in this situation
#     :param offset: Determines whether to add NaN values during the initial window of calculations
#     :return: Predicted Y values as pd.Series
#     """
#     rolling_full = PandasRollingOLS(y=y, x=x, window=window, has_const=has_const, use_const=use_const)
#     rolling = pd.DataFrame(rolling_full.predicted)
#     rolling = rolling.iloc[window::window, :].values
#     rolling = pd.Series(np.concatenate(rolling).ravel(), name='predicted')
#
#     pad = []
#     if offset:
#         for i in range(window - 1):
#             pad.append(np.nan)
#
#         pad = pd.DataFrame(pd.Series(pad, name="predicted"))
#         rolling = pd.concat([pad, rolling]).reset_index(drop=True)
#
#     return rolling_full.predicted


