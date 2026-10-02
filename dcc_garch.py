"""DCC-GARCH (Engle, 2002): multi-asset volatility with time-varying correlation.

Two-step estimation:
  1. Fit a univariate GARCH(1,1) to each asset and standardize its returns.
  2. Fit the correlation dynamics (a, b) on the standardized residuals.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

Z_SCORES = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}


def _garch_variance(r, omega, alpha, beta, s0):
    """GARCH(1,1) variance path; day 0 is s0, then s2[t] = omega + alpha*r[t-1]^2 + beta*s2[t-1]."""
    s2 = np.empty(len(r))
    s2[0] = s0
    if len(r) > 1:
        x = omega + alpha * r[:-1] ** 2
        s2[1:] = lfilter([1.0], [1.0, -beta], x, zi=[beta * s0])[0]
    return s2


def fit_garch11(r):
    """Maximum-likelihood GARCH(1,1) for one return series (zero-mean convention)."""
    r = np.asarray(r, dtype=float)
    scale = 100.0  # fit in percent units for better optimizer conditioning
    rp = r * scale
    s0 = rp.var()

    def nll(p):
        omega, alpha, beta = p
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
            return 1e10
        s2 = _garch_variance(rp, omega, alpha, beta, s0)
        return 0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + rp**2 / s2)

    res = minimize(nll, [0.05 * s0, 0.08, 0.90], method="L-BFGS-B", bounds=[(1e-8, None), (0, 1), (0, 1)])
    omega, alpha, beta = res.x
    return {"omega": omega / scale**2, "alpha": alpha, "beta": beta, "s0": s0 / scale**2}


def _dcc_q(z, a, b, q_bar):
    """Q_t = (1-a-b) Q_bar + a z_{t-1} z_{t-1}' + b Q_{t-1}, computed with a linear filter."""
    t_len = z.shape[0]
    zz = z[:, :, None] * z[:, None, :]
    q = np.empty_like(zz)
    q[0] = q_bar
    if t_len > 1:
        x = (1 - a - b) * q_bar + a * zz[:-1]
        q[1:] = lfilter([1.0], [1.0, -b], x, axis=0, zi=(b * q_bar)[None, :, :])[0]
    return q


def _to_corr(q):
    d = 1.0 / np.sqrt(np.einsum("tii->ti", q))
    return q * d[:, :, None] * d[:, None, :]


def _dcc_nll(params, z, q_bar):
    a, b = params
    if a < 0 or b < 0 or a + b >= 0.9999:
        return 1e10
    r = _to_corr(_dcc_q(z, a, b, q_bar))
    sign, logdet = np.linalg.slogdet(r)
    if np.any(sign <= 0):
        return 1e10
    sol = np.linalg.solve(r, z[:, :, None])[:, :, 0]
    quad = np.einsum("ti,ti->t", z, sol)
    return 0.5 * np.sum(logdet + quad - np.einsum("ti,ti->t", z, z))


def fit_dcc_garch(returns):
    """Fit per-asset GARCH(1,1) models, then the DCC parameters (a, b). returns: (T, n) array."""
    returns = np.asarray(returns, dtype=float)
    t_len, n = returns.shape
    garch = [fit_garch11(returns[:, i]) for i in range(n)]
    s2 = np.column_stack(
        [_garch_variance(returns[:, i], g["omega"], g["alpha"], g["beta"], g["s0"]) for i, g in enumerate(garch)]
    )
    z = returns / np.sqrt(s2)
    q_bar = z.T @ z / t_len
    res = minimize(_dcc_nll, [0.03, 0.94], args=(z, q_bar), method="L-BFGS-B", bounds=[(0, 1), (0, 1)])
    a, b = res.x
    return {"garch": garch, "a": a, "b": b, "q_bar": q_bar}


def dcc_filter(returns, model):
    """Conditional volatilities (T, n) and correlation matrices (T, n, n) for every day.

    Row t uses only information up to day t-1, so it is a genuine one-day-ahead forecast.
    """
    returns = np.asarray(returns, dtype=float)
    s2 = np.column_stack(
        [_garch_variance(returns[:, i], g["omega"], g["alpha"], g["beta"], g["s0"]) for i, g in enumerate(model["garch"])]
    )
    z = returns / np.sqrt(s2)
    q = _dcc_q(z, model["a"], model["b"], model["q_bar"])
    return np.sqrt(s2), _to_corr(q)


def forecast_next(returns, model):
    """One-day-ahead (volatilities, correlation matrix) for the day after the last observation."""
    returns = np.asarray(returns, dtype=float)
    s2 = np.column_stack(
        [_garch_variance(returns[:, i], g["omega"], g["alpha"], g["beta"], g["s0"]) for i, g in enumerate(model["garch"])]
    )
    z = returns / np.sqrt(s2)
    q = _dcc_q(z, model["a"], model["b"], model["q_bar"])
    next_s2 = np.array(
        [g["omega"] + g["alpha"] * returns[-1, i] ** 2 + g["beta"] * s2[-1, i] for i, g in enumerate(model["garch"])]
    )
    a, b = model["a"], model["b"]
    q_next = (1 - a - b) * model["q_bar"] + a * np.outer(z[-1], z[-1]) + b * q[-1]
    return np.sqrt(next_s2), _to_corr(q_next[None])[0]


def constant_correlation(model):
    """The CCC-GARCH correlation matrix: Q_bar rescaled to a correlation matrix."""
    return _to_corr(model["q_bar"][None])[0]


def portfolio_volatility(weights, sigma, corr):
    """sqrt(w' D R D w). Works for one day (n,), (n, n) or a path (T, n), (T, n, n)."""
    v = np.asarray(weights) * sigma
    if corr.ndim == 2 and v.ndim == 2:
        corr = np.broadcast_to(corr, (v.shape[0],) + corr.shape)
    return np.sqrt(np.einsum("...i,...ij,...j->...", v, corr, v))


def portfolio_var(weights, sigma, corr, confidence=0.95, portfolio_value=1.0):
    """Parametric portfolio VaR with zero mean (RiskMetrics convention)."""
    if confidence not in Z_SCORES:
        raise ValueError(f"confidence must be one of {list(Z_SCORES)}")
    return Z_SCORES[confidence] * portfolio_volatility(weights, sigma, corr) * portfolio_value


def simulate_regime_returns(seed=0):
    """3 assets: calm (rho=0.2, vol 1%) -> crisis (rho=0.85, vol 2.5%) -> calm again."""
    rng = np.random.default_rng(seed)
    n = 3
    regimes = [(300, 0.010, 0.20), (100, 0.025, 0.85), (300, 0.010, 0.20)]
    rets, true_rho = [], []
    for days, vol, rho in regimes:
        corr = np.full((n, n), rho)
        np.fill_diagonal(corr, 1.0)
        rets.append(rng.multivariate_normal(np.zeros(n), corr * vol**2, size=days))
        true_rho.append(np.full(days, rho))
    return np.vstack(rets), np.concatenate(true_rho)


def plot_synthetic_recovery(est_rho, true_rho, const_rho, out_path="dcc_synthetic_recovery.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(true_rho, color="black", linewidth=2, label="True correlation")
    ax.plot(est_rho, color="#C44E52", linewidth=1.8, label="DCC-GARCH estimate")
    ax.axhline(const_rho, color="#4C72B0", linestyle="--", linewidth=2, label=f"Constant correlation (CCC) = {const_rho:.2f}")
    ax.axvspan(300, 400, color="gray", alpha=0.15)
    ax.text(305, 0.02, "crisis regime", fontsize=9)
    ax.set_ylim(-0.1, 1)
    ax.set_xlabel("Day")
    ax.set_ylabel("Correlation between asset 1 and asset 2")
    ax.set_title("DCC-GARCH Tracks a Correlation Jump That a Constant Model Misses")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart to '{out_path}'")


if __name__ == "__main__":
    returns, true_rho = simulate_regime_returns()
    model = fit_dcc_garch(returns)
    sigma, corr_path = dcc_filter(returns, model)
    est_rho = corr_path[:, 0, 1]
    const_rho = constant_correlation(model)[0, 1]

    print(f"DCC parameters: a={model['a']:.4f}, b={model['b']:.4f}, a+b={model['a'] + model['b']:.4f}")
    for i, g in enumerate(model["garch"]):
        print(f"Asset {i + 1} GARCH(1,1): alpha={g['alpha']:.4f}, beta={g['beta']:.4f}")
    print()
    print(f"{'Period':<22} {'True rho':>9} {'DCC estimate':>13} {'Constant (CCC)':>15}")
    for name, sl in [("Calm (days 100-300)", slice(100, 300)), ("Crisis (days 320-400)", slice(320, 400)), ("Calm again (days 500-700)", slice(500, 700))]:
        print(f"{name:<22} {true_rho[sl].mean():>9.2f} {est_rho[sl].mean():>13.2f} {const_rho:>15.2f}")

    weights = np.ones(3) / 3
    tomorrow_sigma, tomorrow_corr = forecast_next(returns, model)
    print(f"\nTomorrow's equal-weight portfolio 95% VaR (per 1.0 of value): "
          f"{portfolio_var(weights, tomorrow_sigma, tomorrow_corr, 0.95):.4%}")

    plot_synthetic_recovery(est_rho, true_rho, const_rho)
