"""Independent verification of this repo's GARCH(1,1) (dcc_garch.fit_garch11) against the `arch` package.

Tests (acceptance criteria are fixed in TOL before looking at results):
  A1  Recursion and likelihood as originally specified: same parameters, backcast = our initial variance
  A2  Same as A1 after aligning the initial-variance convention (see CONVENTION below)
  B   Estimation: do both optimizers reach the same parameters and an equally good likelihood?
  C   Initialization sensitivity: effect of arch's default backcast versus ours
  D   One-step-ahead variance forecast
  E   Simulation with known parameters (repeated), compared to the truth
  F   Real data (SPY, TLT, GLD)

CONVENTION: this repo starts the recursion at sigma2_0 = s0 (the sample variance). arch treats the
backcast b as the pre-sample squared residual and variance, so its first-day variance is
omega + (alpha + beta) * b. A1 assumes the two are the same; A2 aligns them.

Usage:
    pip install arch
    python validate_garch_vs_arch.py --end 2026-10-01
"""

import argparse

import numpy as np
from arch import arch_model

from dcc_garch import _garch_variance, fit_garch11

SCALE = 100.0  # both implementations are compared in percent units
TOL = {
    "path_rel": 1e-8,        # A: max relative difference of the variance path
    "loglik_abs": 1e-6,      # A: absolute log-likelihood difference
    "alpha_abs": 0.005,      # B/F: |alpha_ours - alpha_arch|
    "beta_abs": 0.010,       # B/F: |beta_ours - beta_arch|
    "persist_abs": 0.005,    # B/F: |(alpha+beta)_ours - (alpha+beta)_arch|
    "loglik_better": 0.1,    # B/F: ours may be at most this much below arch (flat likelihood tolerance)
    "forecast_rel": 1e-8,    # D: relative difference of the 1-day variance forecast
    "sim_ratio": 0.25,       # E: mean |ours - arch| must be below this fraction of the RMSE against the truth
}


def variance_and_loglik(r_pct, omega, alpha, beta, s0, aligned=False):
    """Our recursion. aligned=True starts at omega + (alpha+beta)*s0 (arch's convention) instead of s0."""
    start = omega + (alpha + beta) * s0 if aligned else s0
    s2 = _garch_variance(r_pct, omega, alpha, beta, start)
    loglik = -0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + r_pct**2 / s2)
    return s2, loglik


def fit_ours(r):
    """Our estimator; returns parameters in percent units plus the initial variance it used."""
    p = fit_garch11(r)
    return {"omega": p["omega"] * SCALE**2, "alpha": p["alpha"], "beta": p["beta"], "s0": p["s0"] * SCALE**2}


def fit_arch(r_pct, backcast=None):
    model = arch_model(r_pct, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    return model.fit(disp="off", backcast=backcast)


def arch_params(res):
    return {"omega": float(res.params["omega"]), "alpha": float(res.params["alpha[1]"]), "beta": float(res.params["beta[1]"])}


def status(ok):
    return "PASS" if ok else "FAIL"


def compare_one(label, r, results):
    """Run tests A-D on one return series (decimal returns)."""
    r_pct = np.asarray(r) * SCALE
    s0 = float(r_pct.var())
    ours = fit_ours(r)
    arch_bc = fit_arch(r_pct, backcast=s0)
    arch_def = fit_arch(r_pct)
    pa, pd_ = arch_params(arch_bc), arch_params(arch_def)
    s2_arch = np.asarray(arch_bc.conditional_volatility) ** 2
    ll_arch = float(arch_bc.loglikelihood)

    # A1: as originally specified (our recursion starts at s0)
    s2_a1, ll_a1 = variance_and_loglik(r_pct, pa["omega"], pa["alpha"], pa["beta"], s0)
    a1_path, a1_ll = float(np.max(np.abs(s2_a1 / s2_arch - 1))), abs(ll_a1 - ll_arch)
    first_day = float(abs(s2_a1[0] / s2_arch[0] - 1))
    # A2: aligned initial-variance convention
    s2_a2, ll_a2 = variance_and_loglik(r_pct, pa["omega"], pa["alpha"], pa["beta"], s0, aligned=True)
    a2_path, a2_ll = float(np.max(np.abs(s2_a2 / s2_arch - 1))), abs(ll_a2 - ll_arch)

    # B: each optimizer's answer; likelihood compared under arch's convention (the aligned objective)
    d_alpha, d_beta = ours["alpha"] - pa["alpha"], ours["beta"] - pa["beta"]
    d_persist = (ours["alpha"] + ours["beta"]) - (pa["alpha"] + pa["beta"])
    _, ll_ours_as_impl = variance_and_loglik(r_pct, ours["omega"], ours["alpha"], ours["beta"], s0)
    _, ll_ours_aligned = variance_and_loglik(r_pct, ours["omega"], ours["alpha"], ours["beta"], s0, aligned=True)
    gap_as_impl, gap_aligned = ll_ours_as_impl - ll_arch, ll_ours_aligned - ll_arch

    # C: arch default backcast vs backcast = our s0
    d_alpha_c, d_beta_c = pd_["alpha"] - pa["alpha"], pd_["beta"] - pa["beta"]
    vol_def, vol_bc = np.asarray(arch_def.conditional_volatility), np.asarray(arch_bc.conditional_volatility)
    rel = np.abs(vol_def / vol_bc - 1)

    # D: next-day variance forecast, same parameters and aligned convention
    fc_arch = float(arch_bc.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
    fc_ours = pa["omega"] + pa["alpha"] * r_pct[-1] ** 2 + pa["beta"] * s2_a2[-1]
    fc_rel = abs(fc_ours / fc_arch - 1)

    print(f"--- {label} (n = {len(r_pct)}) ---")
    print(f"  Parameters (percent units)   omega        alpha     beta      alpha+beta   arch-convention log-likelihood")
    print(f"  ours                         {ours['omega']:<12.6f} {ours['alpha']:<9.4f} {ours['beta']:<9.4f} {ours['alpha'] + ours['beta']:<12.4f} {ll_ours_aligned:.4f}")
    print(f"  arch (backcast = our s0)     {pa['omega']:<12.6f} {pa['alpha']:<9.4f} {pa['beta']:<9.4f} {pa['alpha'] + pa['beta']:<12.4f} {ll_arch:.4f}")
    print(f"  arch (default backcast)      {pd_['omega']:<12.6f} {pd_['alpha']:<9.4f} {pd_['beta']:<9.4f} {pd_['alpha'] + pd_['beta']:<12.4f} {float(arch_def.loglikelihood):.4f}")
    print(f"  A1 as specified: max relative variance diff {a1_path:.2e} (first day {first_day:.2e}), log-likelihood diff {a1_ll:.2e}")
    print(f"  A2 aligned init: max relative variance diff {a2_path:.2e}, log-likelihood diff {a2_ll:.2e}")
    print(f"  B estimation: d_alpha {d_alpha:+.4f}, d_beta {d_beta:+.4f}, d_persistence {d_persist:+.4f}; "
          f"log-likelihood ours minus arch: {gap_aligned:+.5f} (arch convention), {gap_as_impl:+.5f} (our own convention)")
    print(f"  C initialization (arch default vs our backcast): d_alpha {d_alpha_c:+.4f}, d_beta {d_beta_c:+.4f}, "
          f"volatility path diff max {rel.max():.2%}, mean {rel.mean():.3%}, last day {rel[-1]:.4%}")
    print(f"  D forecast: relative difference {fc_rel:.2e}\n")

    results.append({
        "label": label,
        "A1": a1_path < TOL["path_rel"] and a1_ll < TOL["loglik_abs"],
        "A2": a2_path < TOL["path_rel"] and a2_ll < TOL["loglik_abs"],
        "B": abs(d_alpha) < TOL["alpha_abs"] and abs(d_beta) < TOL["beta_abs"]
        and abs(d_persist) < TOL["persist_abs"] and gap_aligned > -TOL["loglik_better"],
        "D": fc_rel < TOL["forecast_rel"],
    })


def simulate_garch(n, omega, alpha, beta, rng, burn=500):
    total = n + burn
    s2 = omega / (1 - alpha - beta)
    e = rng.standard_normal(total)
    r = np.empty(total)
    for t in range(total):
        r[t] = np.sqrt(s2) * e[t]
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return r[burn:]


def simulation_test(n_reps=100, n_obs=1500, truth=(0.02, 0.08, 0.90), seed=1):
    omega_t, alpha_t, beta_t = truth
    rng = np.random.default_rng(seed)
    ours_est, arch_est, gaps = [], [], []
    for _ in range(n_reps):
        r_pct = simulate_garch(n_obs, omega_t, alpha_t, beta_t, rng)
        s0 = float(r_pct.var())
        o = fit_ours(r_pct / SCALE)
        res = fit_arch(r_pct, backcast=s0)
        a = arch_params(res)
        _, ll_o = variance_and_loglik(r_pct, o["omega"], o["alpha"], o["beta"], s0, aligned=True)
        ours_est.append([o["omega"], o["alpha"], o["beta"]])
        arch_est.append([a["omega"], a["alpha"], a["beta"]])
        gaps.append(ll_o - float(res.loglikelihood))
    ours_est, arch_est, gaps = np.array(ours_est), np.array(arch_est), np.array(gaps)
    truth_vec = np.array(truth)

    print(f"--- E: simulation, {n_reps} repetitions of n = {n_obs}, truth omega={omega_t}, alpha={alpha_t}, beta={beta_t} ---")
    print(f"  {'':<8} {'bias ours':>10} {'bias arch':>10} {'RMSE ours':>10} {'RMSE arch':>10} {'mean |ours-arch|':>17} {'max |ours-arch|':>16}")
    ok = True
    for k, name in enumerate(["omega", "alpha", "beta"]):
        rmse_o = np.sqrt(np.mean((ours_est[:, k] - truth_vec[k]) ** 2))
        rmse_a = np.sqrt(np.mean((arch_est[:, k] - truth_vec[k]) ** 2))
        diff = np.abs(ours_est[:, k] - arch_est[:, k])
        print(f"  {name:<8} {np.mean(ours_est[:, k] - truth_vec[k]):>+10.4f} {np.mean(arch_est[:, k] - truth_vec[k]):>+10.4f} "
              f"{rmse_o:>10.4f} {rmse_a:>10.4f} {np.mean(diff):>17.5f} {np.max(diff):>16.4f}")
        ok = ok and np.mean(diff) < TOL["sim_ratio"] * max(rmse_o, rmse_a)
    print(f"  log-likelihood ours minus arch (arch convention): mean {gaps.mean():+.5f}, min {gaps.min():+.4f}, "
          f"max {gaps.max():+.4f}; ours below arch by more than 0.01 in {int((gaps < -0.01).sum())} of {n_reps} runs\n")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Verify GARCH(1,1) against the arch package")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None, help="exclusive end date; fix it for reproducible results")
    parser.add_argument("--reps", type=int, default=100)
    args = parser.parse_args()

    import arch as arch_pkg

    from dcc_backtest_data import load_returns

    print(f"arch version {arch_pkg.__version__}\n")
    results = []
    tickers = ["SPY", "TLT", "GLD"]
    df = load_returns(tickers, args.start, args.end)
    print(f"Real data: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)} days)\n")
    for t in tickers:
        compare_one(f"F: {t}", df[t].values, results)

    sim_ok = simulation_test(args.reps)

    print("=== Summary against pre-set acceptance criteria ===")
    print(f"{'Series':<10} {'A1 as specified':>16} {'A2 aligned init':>16} {'B estimation':>13} {'D forecast':>11}")
    for r in results:
        print(f"{r['label']:<10} {status(r['A1']):>16} {status(r['A2']):>16} {status(r['B']):>13} {status(r['D']):>11}")
    print(f"E simulation recovery and agreement: {status(sim_ok)}")
    print("Tolerances:", ", ".join(f"{k}={v}" for k, v in TOL.items()))


if __name__ == "__main__":
    main()
