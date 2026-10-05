"""Stock/bond correlation across regimes: DCC-GARCH on SPY (stocks) and TLT (long Treasuries).

Usage:
    python stock_bond_regimes.py
    python stock_bond_regimes.py --start 2003-01-01
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dcc_backtest_data import load_returns
from dcc_garch import constant_correlation, dcc_filter, fit_dcc_garch


def yearly_table(df, dcc_corr):
    """Per calendar year: mean DCC correlation, in-year sample correlation, and annual returns."""
    daily = df.copy()
    daily["dcc"] = dcc_corr
    daily["mix"] = 0.6 * daily["SPY"] + 0.4 * daily["TLT"]
    rows = []
    for year, g in daily.groupby(daily.index.year):
        rows.append(
            {
                "year": year,
                "days": len(g),
                "dcc_corr": g["dcc"].mean(),
                "sample_corr": g["SPY"].corr(g["TLT"]),
                "SPY": (1 + g["SPY"]).prod() - 1,
                "TLT": (1 + g["TLT"]).prod() - 1,
                "60/40": (1 + g["mix"]).prod() - 1,
            }
        )
    return pd.DataFrame(rows).set_index("year")


def plot_regimes(df, dcc_corr, const, table, out_path="stock_bond_correlation.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [1.2, 1]})

    ax1.plot(df.index, dcc_corr, color="#C44E52", linewidth=1.2, label="DCC-GARCH correlation")
    ax1.axhline(const, color="#4C72B0", linestyle="--", linewidth=1.8, label=f"Full-sample constant = {const:+.2f}")
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"), color="gray", alpha=0.2)
    ax1.text(pd.Timestamp("2022-01-15"), ax1.get_ylim()[0] * 0.9 if ax1.get_ylim()[0] < 0 else -0.8, "2022", fontsize=9)
    ax1.set_ylabel("SPY / TLT correlation")
    ax1.set_title("Stock/Bond Correlation Is Not Constant, and Not Always Negative")
    ax1.legend(loc="upper left")

    years = table.index.astype(int)
    x = np.arange(len(years))
    width = 0.4
    ax2.bar(x - width / 2, table["SPY"] * 100, width, color="#4C72B0", label="SPY (stocks)")
    ax2.bar(x + width / 2, table["TLT"] * 100, width, color="#DD8452", label="TLT (long Treasuries)")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(y) for y in years], rotation=60, fontsize=8)
    ax2.set_ylabel("Calendar-year return (%)")
    ax2.set_title("Calendar-Year Returns: 2022 Is the Year Both Fell Together")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart to '{out_path}'")


def main():
    parser = argparse.ArgumentParser(description="Stock/bond correlation regimes via DCC-GARCH")
    parser.add_argument("--start", default="2003-01-01")
    args = parser.parse_args()

    df = load_returns(["SPY", "TLT"], args.start)
    print(f"Data: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)} days)\n")

    model = fit_dcc_garch(df.values)
    _, corr = dcc_filter(df.values, model)
    dcc_corr = pd.Series(corr[:, 0, 1], index=df.index)
    const = constant_correlation(model)[0, 1]
    print(f"DCC parameters: a={model['a']:.4f}, b={model['b']:.4f}; constant correlation = {const:+.3f}\n")

    table = yearly_table(df, dcc_corr)
    print(f"{'Year':<6} {'Days':>5} {'DCC corr':>9} {'Sample corr':>12} {'SPY':>8} {'TLT':>8} {'60/40':>8}")
    for year, r in table.iterrows():
        print(f"{year:<6} {int(r['days']):>5} {r['dcc_corr']:>+9.2f} {r['sample_corr']:>+12.2f} "
              f"{r['SPY']:>8.1%} {r['TLT']:>8.1%} {r['60/40']:>8.1%}")

    neg = (table["sample_corr"] < 0).sum()
    print(f"\nYears with negative in-year correlation: {neg} of {len(table)}")
    print(f"Years where SPY and TLT both fell: {list(table.index[(table['SPY'] < 0) & (table['TLT'] < 0)])}")

    plot_regimes(df, dcc_corr, const, table)


if __name__ == "__main__":
    main()
