"""
.. module:: squeeze
   :synopsis: Squeeze Related Indicators.

.. moduleauthor:: Dave Raring (dlraring)

"""
import numpy as np
import pandas as pd
from IPython.core.display import display

from ta.momentum import ROCIndicator, SMOIndicator
from ta.trend import SMAIndicator, MACD
from ta.utils import IndicatorMixin, sma, pulse_conversion, ema
from ta.volatility import BollingerBands, KeltnerChannel, AverageTrueRange


def _calc_oscillator_color(row):
    color = '#000000'
    if row['smo_shift'] > row['smo']:
        if row['smo'] >= 0:
            color = '#00ffff'
        else:
            color = '#cc00cc'
    elif row['smo'] >= 0:
        color = '#009b9b'
    else:
        color = '#ff9bff'

    row['osc_color'] = color
    return row


# for i in range(len(self._oscillator)):
#     if i < len(self._oscillator) - 1:
#         if self._oscillator[i+1] > self._oscillator[i]:
#             if self._oscillator[i] >= 0:
#                 osc_color[i] = '#00ffff'
#             else:
#                 osc_color[i] = '#cc00cc'
#         elif self._oscillator[i] >= 0:
#             osc_color[i] = '#009b9b'
#         else:
#             osc_color[i] = '#ff9bff'

class TtmSqueeze(IndicatorMixin):
    """TTM Squeeze

    This indicator is designed to show when price becomes compressed and then the potential
    direction in which the price will go when the compression resolves.  Bollinger Bands AND
    Keltner Channel define the market conditions, i.e. when BB is narrower than KC then we have
    a market squeeze. When BB break Outside the KC then trade in the direction of the smoothed
    Momentum(12)

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr

    Args:
        high(pandas.Series):    dataset 'High' column.
        low(pandas.Series):     dataset 'Low' column.
        close(pandas.Series):   dataset 'Close' column.
        chan_period(int):       Bollinger Bands AND Keltner Channel length
        bol_band_std_dev(float):  width of the Bollinger Bands
        kelt_std_dev(float):      width of the Keltner Bands
        mom_period(int):        number of bars for momentum indicator
        mom_ema(int):           EMA of momentum
        fillna(bool):           if True, fill nan values.
    """

    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, chan_period: int = 14,
                 bol_band_std_dev: int = 2, kelt_std_dev: int = 1, mom_period: int = 13,
                 mom_ema: int = 21, fillna: bool = False):
        self._high = high
        self._low = low
        self._close = close
        self._chan_period = chan_period
        self._bol_band_std_dev = bol_band_std_dev
        self._kelt_std_dev = kelt_std_dev
        self._mom_period = mom_period
        self._mom_ema = mom_ema
        self._fillna = fillna
        self._run()

    def _run(self):
        # Bollinger Bands
        bb_indicator = BollingerBands(close=self._close, n=self._chan_period, ndev=self._bol_band_std_dev,
                                      fillna=self._fillna)
        self._bb_hband = bb_indicator.bollinger_hband()
        self._bb_lband = bb_indicator.bollinger_lband()

        # Keltner Channel
        kb_indicator = KeltnerChannel(high=self._high, low=self._low, close=self._close, n=self._chan_period,
                                      n_atr=self._kelt_std_dev, fillna=self._fillna, ov=False)
        self._kb_hband = kb_indicator.keltner_channel_hband()
        self._kb_lband = kb_indicator.keltner_channel_lband()

        # Momentum Oscillator
        smo_indicator = SMOIndicator(high=self._high, low=self._low, close=self._close, n=self._mom_period,
                                     fillna=self._fillna)
        self._oscillator = smo_indicator.smo()

        # Bar and Signal Colors
        self._squeeze = bb_indicator.bollinger_wband() - kb_indicator.keltner_channel_wband()
        self._squeeze = self._squeeze.ge(0).astype(int)

    def oscillator(self) -> pd.Series:
        tp = self._oscillator
        return pd.Series(tp, name='oscillator')

    def oscillator_color(self) -> pd.Series:
        temp_df = pd.DataFrame(self._oscillator)
        temp_df['smo_shift'] = temp_df['smo'].shift(-1).fillna(method='ffill')
        temp_df = temp_df.apply(_calc_oscillator_color, axis=1)
        return temp_df['osc_color']

    def squeeze(self):
        tp = self._squeeze
        return pd.Series(tp, name="squeeze")

    def squeeze_color(self):
        sqz_color = self._squeeze.apply(lambda x: 'green' if x else 'red')
        return pd.Series(sqz_color, name="squeeze_color")
