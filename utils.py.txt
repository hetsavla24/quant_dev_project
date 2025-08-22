import numpy as np
import matplotlib.pyplot as plt

def sharpe_ratio(returns, periods_per_year=252*6.5*60/5):  # approx number of 5-min periods in a year
    r = np.array(returns)
    if r.std() == 0:
        return 0.0
    return (r.mean() / r.std()) * np.sqrt(periods_per_year)

def equity_curve_from_trades(trades, starting_capital=200000.0):
    pnl = trades['pnl'].fillna(0.0).cumsum()
    equity = starting_capital + pnl
    return equity

def max_drawdown(equity):
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return dd.min(), dd

def save_equity_and_drawdown(trades, out_dir, starting_capital=200000.0):
    equity = equity_curve_from_trades(trades, starting_capital)
    mdd, dd_series = max_drawdown(equity)

    plt.figure()
    equity.plot()
    plt.title('Equity Curve')
    plt.xlabel('Trade #')
    plt.ylabel('Equity')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/equity_curve.png")
    plt.close()

    plt.figure()
    dd_series.plot()
    plt.title('Drawdown')
    plt.xlabel('Trade #')
    plt.ylabel('Drawdown')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/drawdown.png")
    plt.close()

    return float(mdd)
