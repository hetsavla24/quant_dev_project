import numpy as np
import pandas as pd

def indicator_to_signal(df):
    """Convert indicators into Buy(1), Sell(-1), Hold(0) signals"""
    signals = pd.DataFrame(index=df.index)

    # RSI → overbought (>70 = sell), oversold (<30 = buy)
    signals['RSI_sig'] = np.where(df['rsi_14'] > 70, -1, np.where(df['rsi_14'] < 30, 1, 0))

    # MACD → bullish if MACD > signal, bearish if MACD < signal
    signals['MACD_sig'] = np.where(df['macd'] > df['macd_sig'], 1, -1)

    # Bollinger → price > upper band = sell, price < lower band = buy
    signals['BB_sig'] = np.where(df['close'] > df['bb_up'], -1,
                                np.where(df['close'] < df['bb_lo'], 1, 0))

    # SuperTrend → close > supertrend = buy, else sell
    signals['ST_sig'] = np.where(df['close'] > df['SuperTrend'], 1, -1)

    # Stochastic → %K > %D bullish, else bearish
    signals['STOCH_sig'] = np.where(df['STOCH_K'] > df['STOCH_D'], 1, -1)

    # ADX → trend strength (ignore if <20). Here just trend direction.
    signals['ADX_sig'] = np.where(df['ADX'] > 20,
                                 np.where(df['+dm'] > df['-dm'], 1, -1),
                                 0)

    return signals

def composite_signal(df, method="vote"):
    """Combine signals into one composite Buy/Sell/Hold"""
    signals = indicator_to_signal(df)

    if method == "vote":
        # majority voting
        comp = signals.sum(axis=1)
        df['composite_signal'] = np.where(comp > 0, 1, np.where(comp < 0, -1, 0))
    elif method == "weighted":
        # give more weight to MACD & SuperTrend
        weights = {"RSI_sig":1, "MACD_sig":2, "BB_sig":1, "ST_sig":2, "STOCH_sig":1, "ADX_sig":1}
        comp = sum(signals[col]*w for col,w in weights.items())
        df['composite_signal'] = np.where(comp > 0, 1, np.where(comp < 0, -1, 0))

    return df
