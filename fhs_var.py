"""Filtered Historical Simulation (FHS) on top of DCC-GARCH.

Instead of assuming normal shocks (VaR = z * sigma_p), FHS:
  1. filters history with the model so each day's shock is divided by its own volatility,
  2. keeps those standardized shocks as an empirical distribution (fat tails included),
  3. rescales them by tomorrow's forecast volatility.
"""

import numpy as np

from dcc_garch import dcc_filter, forecast_next, portfolio_volatility


def standardized_portfolio_shocks(returns, weights, model):
    """Portfolio return divided by the DCC-implied portfolio volatility, for every day.

    Returns (shocks, sigma_p). Row t only uses information up to day t-1 for sigma_p.
    """
    returns = np.asarray(returns, dtype=float)
    sigma, corr = dcc_filter(returns, model)
    sigma_p = portfolio_volatility(weights, sigma, corr)
    return (returns @ np.asarray(weights)) / sigma_p, sigma_p


def fhs_var_from_shocks(shocks, sigma_p_next, confidence=0.95, portfolio_value=1.0):
    """VaR from the empirical lower quantile of standardized shocks, scaled by tomorrow's volatility."""
    q = np.nanquantile(shocks, 1 - confidence)
    return -q * sigma_p_next * portfolio_value


def fhs_simulate_portfolio_returns(returns, weights, model, n_sim=100_000, seed=None):
    """Full multivariate FHS: bootstrap whole days of decorrelated shocks and rebuild tomorrow's returns.

    1. z_t = r_t / sigma_t             remove volatility
    2. e_t = L_t^-1 z_t                remove correlation (R_t = L_t L_t')
    3. resample e* from {e_t}          i.i.d. shocks with fat tails and joint extremes preserved
    4. r* = sigma_next * (L_next e*)   put tomorrow's volatility and correlation back
    Works for any weights, so the same scenarios can be reused for what-if portfolios.
    """
    returns = np.asarray(returns, dtype=float)
    sigma, corr = dcc_filter(returns, model)
    z = returns / sigma
    chol = np.linalg.cholesky(corr)
    eps = np.linalg.solve(chol, z[:, :, None])[:, :, 0]

    sigma_next, corr_next = forecast_next(returns, model)
    chol_next = np.linalg.cholesky(corr_next)

    rng = np.random.default_rng(seed)
    draws = eps[rng.integers(0, len(eps), n_sim)]
    sim_returns = (draws @ chol_next.T) * sigma_next
    return sim_returns @ np.asarray(weights)


def fhs_var_simulated(returns, weights, model, confidence=0.95, portfolio_value=1.0, n_sim=100_000, seed=None):
    sim = fhs_simulate_portfolio_returns(returns, weights, model, n_sim, seed)
    return -np.quantile(sim, 1 - confidence) * portfolio_value
