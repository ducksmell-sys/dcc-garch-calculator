"""Backtest DCC-GARCH against simpler covariance models on real multi-asset data.

Compares four ways of estimating next-day portfolio risk:
  Static    sample second-moment matrix of the last `window` days
  EWMA      RiskMetrics exponentially weighted covariance (lambda = 0.94)
  CCC-GARCH GARCH(1,1) volatilities with a constant correlation matrix
  DCC-GARCH GARCH(1,1) volatilities with time-varying correlation

Usage:
    python dcc_backtest_data.py
    python dcc_backtest_data.py --tickers SPY TLT GLD --start 2017-01-01 --confidence 0.99
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from dcc_garch import (
    Z_SCORES,
    constant_correlation,
    dcc_filter,
    fit_dcc_garch,
    forecast_next,
    portfolio_var,
    portfolio_volatility,
)
from risk_tests import christoffersen_independence_test, conditional_coverage_test, kupiec_test

MODELS = ["Static", "EWMA", "CCC-GARCH", "DCC-GARCH"]
COLORS = {"Static": "#8C8C8C", "EWMA": "#DD8452", "CCC-GARCH": "#4C72B0", "DCC-GARCH": "#C44E52"}


def load_returns(tickers, start, end=None):
    import yfinance as yf

    prices = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)["Close"][tickers]
    return prices.pct_change().dropna()


def ewma_covariance_path(returns, window, lam=0.94):
    """S[t] = lam * S[t-1] + (1-lam) * r[t-1] r[t-1]'; row t only uses data before day t."""
    t_len, n = returns.shape
    rr = returns[:, :, None] * returns[:, None, :]
    s0 = returns[:window].T @ returns[:window] / window
    s = np.empty_like(rr)
    s[0] = s0
    x = (1 - lam) * rr[:-1]
    s[1:] = lfilter([1.0], [1.0, -lam], x, axis=0, zi=(lam * s0)[None, :, :])[0]
    return s


def rolling_backtest(returns, weights, window=500, refit_every=20, confidence=0.95):
    """Daily one-step-ahead portfolio volatility from each model; GARCH models refit every `refit_every` days."""
    t_len, n = returns.shape
    n_test = t_len - window
    vols = {m: np.empty(n_test) for m in MODELS}

    ewma_cov = ewma_covariance_path(returns, window)
    for t in range(window, t_len):
        vols["EWMA"][t - window] = np.sqrt(weights @ ewma_cov[t] @ weights)
        hist = returns[t - window : t]
        vols["Static"][t - window] = np.sqrt(weights @ (hist.T @ hist / window) @ weights)

    for t0 in range(window, t_len, refit_every):
        t1 = min(t0 + refit_every, t_len)
        model = fit_dcc_garch(returns[t0 - window : t0])
        sigma, corr = dcc_filter(returns[t0 - window : t1], model)
        sigma, corr = sigma[window:], corr[window:]
        vols["DCC-GARCH"][t0 - window : t1 - window] = portfolio_volatility(weights, sigma, corr)
        vols["CCC-GARCH"][t0 - window : t1 - window] = portfolio_volatility(weights, sigma, constant_correlation(model))

    z = Z_SCORES[confidence]
    var = {m: z * vols[m] for m in MODELS}
    losses = -(returns[window:] @ weights)
    violations = {m: losses > var[m] for m in MODELS}
    return var, losses, violations


def plot_correlations(dates, returns, tickers, out_path="dcc_correlations.png"):
    model = fit_dcc_garch(returns)
    _, corr = dcc_filter(returns, model)
    const = constant_correlation(model)
    pairs = [(i, j) for i in range(len(tickers)) for j in range(i + 1, len(tickers))]

    fig, axes = plt.subplots(len(pairs), 1, figsize=(11, 3.2 * len(pairs)), sharex=True)
    for ax, (i, j) in zip(np.atleast_1d(axes), pairs):
        ax.plot(dates, corr[:, i, j], color="#C44E52", linewidth=1.3, label="DCC-GARCH")
        ax.axhline(const[i, j], color="#4C72B0", linestyle="--", linewidth=1.8, label=f"Constant = {const[i, j]:.2f}")
        ax.axhline(0, color="gray", linewidth=0.6)
        covid = (pd.Timestamp("2020-02-19"), pd.Timestamp("2020-04-07"))
        if dates[0] < covid[0] and dates[-1] > covid[1]:
            ax.axvspan(*covid, color="gray", alpha=0.2)
        ax.set_ylabel(f"{tickers[i]} / {tickers[j]}")
        ax.legend(loc="upper left", fontsize=8)
    np.atleast_1d(axes)[0].set_title("Dynamic Correlation (DCC-GARCH) vs Constant Correlation (grey band = Covid crash)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart to '{out_path}'")


def plot_var_comparison(dates, var, losses, out_path="dcc_var_comparison.png"):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.scatter(dates, losses, s=4, color="lightgray", label="Daily portfolio loss", zorder=1)
    for m in MODELS:
        ax.plot(dates, var[m], color=COLORS[m], linewidth=1.2, label=m, zorder=2)
    top = max(losses.max(), max(v.max() for v in var.values())) * 1.1
    ax.set_ylim(-0.01, top)
    ax.set_ylabel("Loss / 95% VaR (fraction of portfolio)")
    ax.set_title("Portfolio 95% VaR by Model vs Realized Daily Losses")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart to '{out_path}'")


def plot_violation_rates(violations, confidence, out_path="dcc_violation_rates.png"):
    rates = [violations[m].mean() * 100 for m in MODELS]
    pvals = [kupiec_test(violations[m], confidence)["p_value"] for m in MODELS]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(MODELS, rates, color=[COLORS[m] for m in MODELS])
    ax.axhline((1 - confidence) * 100, color="black", linestyle="--", linewidth=1.5, label=f"Expected {(1 - confidence):.0%}")
    for bar, p in zip(bars, pvals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"Kupiec p={p:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Violation rate (%)")
    ax.set_title(f"VaR Violation Rate by Model ({confidence:.0%} VaR)")
    ax.set_ylim(0, max(rates) * 1.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart to '{out_path}'")


def main():
    parser = argparse.ArgumentParser(description="Backtest DCC-GARCH vs simpler models on real data")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "TLT", "GLD"])
    parser.add_argument("--weights", nargs="+", type=float, help="portfolio weights (default: equal)")
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--window", type=int, default=500)
    parser.add_argument("--refit-every", type=int, default=20)
    parser.add_argument("--confidence", type=float, default=0.95, choices=[0.90, 0.95, 0.99])
    parser.add_argument("--portfolio-value", type=float, default=1_000_000)
    args = parser.parse_args()

    tickers = args.tickers
    weights = np.array(args.weights) if args.weights else np.ones(len(tickers)) / len(tickers)
    if len(weights) != len(tickers):
        parser.error("--weights must have one entry per ticker")

    df = load_returns(tickers, args.start)
    returns = df.values
    print(f"Assets: {tickers} | weights: {np.round(weights, 3)}")
    print(f"Data: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)} days), window={args.window}, refit every {args.refit_every} days\n")

    var, losses, violations = rolling_backtest(returns, weights, args.window, args.refit_every, args.confidence)
    test_dates = df.index[args.window :]
    print(f"Backtest: {test_dates[0].date()} ~ {test_dates[-1].date()} ({len(test_dates)} days)\n")

    print(f"{'Model':<11} {'Viol.':>6} {'Rate':>7} {'Kupiec p':>9} {'Indep. p':>9} {'CC p':>8} {'Avg VaR':>9}")
    for m in MODELS:
        k = kupiec_test(violations[m], args.confidence)
        i = christoffersen_independence_test(violations[m])
        c = conditional_coverage_test(violations[m], args.confidence)
        print(f"{m:<11} {k['violations']:>6} {k['rate']:>7.2%} {k['p_value']:>9.4f} {i['p_value']:>9.4f} {c['p_value']:>8.4f} {var[m].mean():>9.3%}")
    print(f"(expected violation rate: {1 - args.confidence:.1%}; p < 0.05 means reject the model)\n")

    model = fit_dcc_garch(returns[-args.window :])
    sigma, corr = forecast_next(returns[-args.window :], model)
    print(f"DCC parameters (last {args.window} days): a={model['a']:.4f}, b={model['b']:.4f}")
    print("Tomorrow's forecast correlations:")
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            print(f"  {tickers[i]}/{tickers[j]}: {corr[i, j]:+.2f} (constant model: {constant_correlation(model)[i, j]:+.2f})")
    print(f"Tomorrow's {args.confidence:.0%} portfolio VaR on {args.portfolio_value:,.0f}: "
          f"{portfolio_var(weights, sigma, corr, args.confidence, args.portfolio_value):,.2f}\n")

    plot_correlations(df.index, returns, tickers)
    plot_var_comparison(test_dates, var, losses)
    plot_violation_rates(violations, args.confidence)


if __name__ == "__main__":
    main()
