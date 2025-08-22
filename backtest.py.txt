import pandas as pd

def run_backtest(spot_df, options_df):
    trades = []

    for i, row in spot_df.iterrows():
        signal = row.get("composite_signal", 0)
        spot_price = row["close"]

        if signal == 0:
            continue  

        expiry = row["closest_expiry"]
        candidates = options_df[options_df["expiry_date"] == expiry]

        if candidates.empty:
            continue  

        # Pick strike closest to spot
        candidates = candidates.copy()
        candidates.loc[:, "dist"] = (candidates["strike_price"] - spot_price).abs()
        atm = candidates.loc[candidates["dist"].idxmin()]

        trades.append({
            "datetime": row["datetime"],
            "signal": signal,
            "instrument": atm["ticker"],
            "strike": atm["strike_price"],
            "entry": atm["open"],
            "exit": atm["close"],
            "pnl": (atm["close"] - atm["open"]) * signal
        })

    return pd.DataFrame(trades)
