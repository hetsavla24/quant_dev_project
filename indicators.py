import pandas as pd
import numpy as np

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(series, period=20, num_std=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, lower

def atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def stochastic(df, k=14, d=3):
    low_min = df['low'].rolling(k).min()
    high_max = df['high'].rolling(k).max()
    stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d

def supertrend(df, period=7, multiplier=3):
    hl2 = (df['high'] + df['low']) / 2
    atr_val = atr(df, period)
    upperband = hl2 + (multiplier * atr_val)
    lowerband = hl2 - (multiplier * atr_val)

    supertrend = pd.Series(index=df.index)
    in_uptrend = True

    for i in range(1, len(df)):
        if df['close'][i] > upperband[i-1]:
            in_uptrend = True
        elif df['close'][i] < lowerband[i-1]:
            in_uptrend = False

        if in_uptrend:
            supertrend[i] = lowerband[i]
        else:
            supertrend[i] = upperband[i]

    return supertrend.ffill()

def adx(df, period=14):
    df = df.copy()
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift()),
                                     abs(df['low'] - df['close'].shift())))
    df['+dm'] = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']),
                         np.maximum(df['high'] - df['high'].shift(), 0), 0)
    df['-dm'] = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()),
                         np.maximum(df['low'].shift() - df['low'], 0), 0)
    
    tr_n = df['tr'].rolling(period).mean()
    plus_di = 100 * (df['+dm'].rolling(period).mean() / tr_n)
    minus_di = 100 * (df['-dm'].rolling(period).mean() / tr_n)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = dx.rolling(period).mean().fillna(0)

    return adx_val, df['+dm'], df['-dm']

def add_basic_indicators(spot_df):
    df = spot_df.copy()
    df['ema_20'] = ema(df['close'], 20)
    df['ema_50'] = ema(df['close'], 50)
    df['rsi_14'] = rsi(df['close'], 14)
    macd_line, sig_line, macd_hist = macd(df['close'])
    df['macd'] = macd_line
    df['macd_sig'] = sig_line
    df['macd_hist'] = macd_hist
    df['bb_up'], df['bb_lo'] = bollinger_bands(df['close'], 20, 2)
    df['atr_14'] = atr(df, 14)
    stoch_k, stoch_d = stochastic(df)
    df['STOCH_K'] = stoch_k
    df['STOCH_D'] = stoch_d
    df['SuperTrend'] = supertrend(df)
    adx_val, plus_dm, minus_dm = adx(df)
    df['ADX'] = adx_val
    df['+dm'] = plus_dm
    df['-dm'] = minus_dm
    return df
