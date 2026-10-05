"""Backtest Filtered Historical Simulation against Gaussian VaR on real multi-asset data.

Four models, each producing a one-day-ahead portfolio VaR:
  Historical     empirical quantile of the last `shock_window` raw portfolio returns (no filtering)
  EWMA-FHS       raw returns divided by EWMA volatility, empirical quantile, rescaled by EWMA volatility
  DCC-Gaussian   normal quantile times DCC-GARCH portfolio volatility
  DCC-FHS        DCC-standardized shocks, empirical quantile, rescaled by DCC-GARCH volatility

Usage:
    python fhs_backtest_data.py
    python fhs_backtest_data.py --tickers SPY TLT --weights 0.6 0.4
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dcc_backtest_data import ewma_covariance_path, load_returns
from dcc_garch import Z_SCORES, dcc_filter, fit_dcc_garch, forecast_next, portfolio_var, portfolio_volatility
from fhs_var import fhs_var_from_shocks, fhs_var_simulated, standardized_portfolio_shocks
from risk_tests import christoffersen_independence_test, conditional_coverage_test, kupiec_test

MODELS = ["Historical", "EWMA-FHS", "DCC-Gaussian", "DCC-FHS"]
COLORS = {"Historical": "#8C8C8C", "EWMA-FHS": "#DD8452", "DCC-Gaussian": "#4C72B0", "DCC-FHS": "#C44E52"}
CONFIDENCES = (0.95, 0.99)
EWMA_INIT = 60


def rolling_backtest_fhs(returns, weights, est_window=500, shock_window=1000, refit_every=20):
    """One-day-ahead VaR for each model and confidence. GARCH/DCC parameters are refit every `refit_every` days
    on the last `est_window` days; shocks are filtered with parameters known before each test day (no look-ahead)."""
    t_len, n = returns.shape
    port = returns @ weights
    start = shock_window
    n_test = t_len - start
    var = {c: {m: np.empty(n_test) for m in MODELS} for c in CONFIDENCES}
    qs = [1 - c for c in CONFIDENCES]

    ewma_cov = ewma_covariance_path(returns, EWMA_INIT)
    ewma_sigma = np.sqrt(np.einsum("i,tij,j->t", weights, ewma_cov, weights))
    ewma_shocks = port / ewma_sigma
    ewma_shocks[:EWMA_INIT] = np.nan

    for t in range(start, t_len):
        i = t - start
        hist_q = np.quantile(port[t - shock_window : t], qs)
        ewma_q = np.nanquantile(ewma_shocks[t - shock_window : t], qs)
        for c, hq, eq in zip(CONFIDENCES, hist_q, ewma_q):
            var[c]["Historical"][i] = -hq
            var[c]["EWMA-FHS"][i] = -eq * ewma_sigma[t]

    for t0 in range(start, t_len, refit_every):
        t1 = min(t0 + refit_every, t_len)
        model = fit_dcc_garch(returns[t0 - est_window : t0])
        block_start = t0 - shock_window
        shocks, sigma_p = standardized_portfolio_shocks(returns[block_start:t1], weights, model)
        for t in range(t0, t1):
            idx = t - block_start
            u_hist = shocks[idx - shock_window : idx]
            q = np.quantile(u_hist, qs)
            for c, quant in zip(CONFIDENCES, q):
                var[c]["DCC-FHS"][t - start] = -quant * sigma_p[idx]
                var[c]["DCC-Gaussian"][t - start] = Z_SCORES[c] * sigma_p[idx]

    losses = -port[start:]
    violations = {c: {m: losses > var[c][m] for m in MODELS} for c in CONFIDENCES}
    return var, losses, violations


def plot_tail_shape(shocks, out_path="fhs_tail_shape.png"):
    q01, q05 = np.quantile(shocks, [0.01, 0.05])
    x = np.linspace(-7, 7, 600)
    pdf = np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax in (ax1, ax2):
        ax.hist(shocks, bins=120, density=True, color="#C44E52", alpha=0.7, label="DCC-standardized portfolio shocks")
        ax.plot(x, pdf, "k--", linewidth=1.8, label="Standard normal")
    ax1.set_xlim(-6, 6)
    ax1.set_title("Whole distribution")
    ax1.legend(loc="upper left", fontsize=8)
    ax2.set_yscale("log")
    ax2.set_xlim(-7, -1)
    ax2.set_ylim(1e-5, 1)
    ax2.axvline(q01, color="#C44E52", linewidth=1.5, label=f"Empirical 1% quantile ({q01:.2f})")
    ax2.axvline(-Z_SCORES[0.99], color="black", linestyle=":", linewidth=1.5, label=f"Normal 1% quantile ({-Z_SCORES[0.99]:.2f})")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_title("Left tail (log scale)")
    fig.suptitle(
        f"Standardized Portfolio Shocks vs Normal: 5% quantile {q05:.2f} (normal {-Z_SCORES[0.95]:.2f}), "
        f"1% quantile {q01:.2f} (normal {-Z_SCORES[0.99]:.2f})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart to '{out_path}'")
    return q01, q05


def plot_var_comparison(dates, var, losses, confidence=0.99, out_path="fhs_var_comparison_99.png"):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.scatter(dates, losses, s=4, color="lightgray", label="Daily portfolio loss", zorder=1)
    for m in MODELS:
        ax.plot(dates, var[confidence][m], color=COLORS[m], linewidth=1.2, label=m, zorder=2)
    top = max(losses.max(), max(v.max() for v in var[confidence].values())) * 1.1
    ax.set_ylim(-0.01, top)
    ax.set_ylabel(f"Loss / {confidence:.0%} VaR (fraction of portfolio)")
    ax.set_title(f"Portfolio {confidence:.0%} VaR by Model vs Realized Daily Losses")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart to '{out_path}'")


def plot_violation_rates(violations, out_path="fhs_violation_rates.png"):
    fig, axes = plt.subplots(1, len(CONFIDENCES), figsize=(12, 5))
    for ax, c in zip(axes, CONFIDENCES):
        rates = [violations[c][m].mean() * 100 for m in MODELS]
        pvals = [kupiec_test(violations[c][m], c)["p_value"] for m in MODELS]
        bars = ax.bar(MODELS, rates, color=[COLORS[m] for m in MODELS])
        ax.axhline((1 - c) * 100, color="black", linestyle="--", linewidth=1.5, label=f"Expected {(1 - c):.0%}")
        for bar, p in zip(bars, pvals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02, f"p={p:.3f}", ha="center", fontsize=8)
        ax.set_ylim(0, max(rates) * 1.2)
        ax.set_title(f"{c:.0%} VaR violation rate (Kupiec p-value above bars)")
        ax.set_ylabel("Violation rate (%)")
        ax.tick_params(axis="x", labelrotation=15)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart to '{out_path}'")


def main():
    parser = argparse.ArgumentParser(description="Backtest Filtered Historical Simulation vs Gaussian VaR")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "TLT", "GLD"])
    parser.add_argument("--weights", nargs="+", type=float, help="portfolio weights (default: equal)")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None, help="exclusive end date; fix it for reproducible results (today's row is intraday)")
    parser.add_argument("--est-window", type=int, default=500, help="days used to fit GARCH/DCC")
    parser.add_argument("--shock-window", type=int, default=1000, help="days of shocks kept for the empirical quantile")
    parser.add_argument("--refit-every", type=int, default=20)
    parser.add_argument("--portfolio-value", type=float, default=1_000_000)
    args = parser.parse_args()

    tickers = args.tickers
    weights = np.array(args.weights) if args.weights else np.ones(len(tickers)) / len(tickers)
    if len(weights) != len(tickers):
        parser.error("--weights must have one entry per ticker")

    df = load_returns(tickers, args.start, args.end)
    returns = df.values
    print(f"Assets: {tickers} | weights: {np.round(weights, 3)}")
    print(f"Data: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)} days)")
    print(f"Fit window {args.est_window}, shock window {args.shock_window}, refit every {args.refit_every} days\n")

    var, losses, violations = rolling_backtest_fhs(returns, weights, args.est_window, args.shock_window, args.refit_every)
    test_dates = df.index[args.shock_window :]
    print(f"Backtest: {test_dates[0].date()} ~ {test_dates[-1].date()} ({len(test_dates)} days)\n")

    for c in CONFIDENCES:
        print(f"--- {c:.0%} VaR (expected violation rate {1 - c:.1%}) ---")
        print(f"{'Model':<13} {'Viol.':>6} {'Rate':>7} {'Kupiec p':>9} {'Indep. p':>9} {'CC p':>8} {'Avg VaR':>9}")
        for m in MODELS:
            k = kupiec_test(violations[c][m], c)
            i = christoffersen_independence_test(violations[c][m])
            cc = conditional_coverage_test(violations[c][m], c)
            print(f"{m:<13} {k['violations']:>6} {k['rate']:>7.2%} {k['p_value']:>9.4f} {i['p_value']:>9.4f} {cc['p_value']:>8.4f} {var[c][m].mean():>9.3%}")
        print()

    model = fit_dcc_garch(returns[-args.est_window :])
    sigma, corr = forecast_next(returns[-args.est_window :], model)
    full_model = fit_dcc_garch(returns)
    shocks, _ = standardized_portfolio_shocks(returns, weights, full_model)
    q01, q05 = plot_tail_shape(shocks)
    print(f"Standardized shock quantiles: 1% = {q01:.2f} (normal {-Z_SCORES[0.99]:.2f}), 5% = {q05:.2f} (normal {-Z_SCORES[0.95]:.2f})\n")

    recent_shocks, _ = standardized_portfolio_shocks(returns[-(args.shock_window + args.est_window) :], weights, model)
    sigma_p_next = portfolio_volatility(weights, sigma, corr)
    print(f"Tomorrow's portfolio VaR on {args.portfolio_value:,.0f}:")
    for c in CONFIDENCES:
        gauss = portfolio_var(weights, sigma, corr, c, args.portfolio_value)
        fhs = fhs_var_from_shocks(recent_shocks[-args.shock_window :], sigma_p_next, c, args.portfolio_value)
        sim = fhs_var_simulated(returns[-args.shock_window :], weights, fit_dcc_garch(returns[-args.est_window :]), c, args.portfolio_value, seed=0)
        print(f"  {c:.0%}: Gaussian {gauss:>10,.2f} | FHS (portfolio-level) {fhs:>10,.2f} | FHS (full simulation) {sim:>10,.2f}")

    plot_var_comparison(test_dates, var, losses)
    plot_violation_rates(violations)


if __name__ == "__main__":
    main()
