# Implementation verification: GARCH(1,1) against the `arch` package

Section of a model validation report for `dcc-garch-calculator`. Scope: the univariate GARCH(1,1)
estimator `dcc_garch.fit_garch11` and its variance recursion. This section verifies that the code
implements the intended model correctly. It does not judge whether the model is appropriate for its
purpose (conceptual soundness and outcomes analysis are covered elsewhere in the repo, in the
backtests and in README_fhs.md).

## 1. Approach

The reference is the independent implementation in `arch` 8.0.0 (same model: zero mean, Gaussian
likelihood, GARCH(1,1), `rescale=False`). The tests are layered so that a failure points to its cause:

| Test | Question | Method |
|---|---|---|
| A | Is the variance recursion and likelihood implemented correctly? | Same parameters, same initial variance, compare the variance path and log-likelihood |
| B | Do the two optimizers reach the same answer? | Compare estimated parameters and the log-likelihood each reaches |
| C | How much does the initialization choice matter? | `arch` default backcast against ours |
| D | Is the one-day-ahead variance forecast the same? | Same parameters, compare the forecast |
| E | Do both recover known parameters? | 100 simulated series (n = 1,500) with omega 0.02, alpha 0.08, beta 0.90 |
| F | Does it hold on real data? | SPY, TLT, GLD daily returns, 2010-01-05 to 2026-09-30 (4,210 days) |

Acceptance criteria were fixed in the script (`TOL` in `validate_garch_vs_arch.py`) before the
first run: path difference below 1e-8 relative and log-likelihood difference below 1e-6 (A);
alpha within 0.005, beta within 0.01, persistence within 0.005 and a log-likelihood no more than
0.1 below arch (B, F); forecast difference below 1e-8 (D); mean estimator disagreement below
25% of the RMSE against the truth (E).

## 2. Result summary

| Series | A1 as specified | A2 aligned initialization | B estimation | D forecast |
|---|---|---|---|---|
| SPY | **FAIL** | PASS | PASS | PASS |
| TLT | **FAIL** | PASS | PASS | PASS |
| GLD | **FAIL** | PASS | PASS | PASS |
| E simulation (100 runs) | | | PASS | |

The first run failed test A as originally specified (A1). The cause was found and is a difference in
initial-variance convention, not an error in the recursion (Finding 1). A2 repeats the test with
the convention aligned. Both are reported because the original specification failed.

## 3. Detailed results

**A. Recursion and likelihood (same parameters = arch's estimates, initial variance b = sample variance)**

| Series | A1: max relative variance difference | A1: log-likelihood difference | A2: max relative variance difference | A2: log-likelihood difference |
|---|---|---|---|---|
| SPY | 2.58e-03 | 3.36e-03 | 2.22e-16 | 0 |
| TLT | 7.47e-04 | 3.78e-04 | 2.22e-16 | 0 |
| GLD | 8.53e-04 | 1.50e-03 | 2.22e-16 | 0 |

In A1 the largest difference is on day 0 for every series and decays afterwards. After aligning the
convention the two paths agree to machine precision.

**B. Estimation (percent units; arch run with the same initial variance)**

| Series | Estimator | omega | alpha | beta | alpha+beta |
|---|---|---|---|---|---|
| SPY | ours | 0.037208 | 0.1503 | 0.8149 | 0.9652 |
| SPY | arch | 0.037210 | 0.1503 | 0.8149 | 0.9652 |
| TLT | ours | 0.013779 | 0.0634 | 0.9202 | 0.9836 |
| TLT | arch | 0.013782 | 0.0634 | 0.9202 | 0.9836 |
| GLD | ours | 0.020445 | 0.0669 | 0.9157 | 0.9826 |
| GLD | arch | 0.020441 | 0.0669 | 0.9157 | 0.9826 |

Parameters agree to the fourth decimal. Under arch's convention our solution's log-likelihood is
within 0.00002 of arch's for every series. Under our own convention the apparent gap is -0.0033
(SPY), -0.0004 (TLT) and -0.0015 (GLD), which closely matches the A1 likelihood difference, so it
comes from the convention and not from the optimizer.

**C. Initialization sensitivity.** With `arch`'s default backcast, alpha and beta move by at most
0.0004. The early-sample volatility path differs by up to 10.4% (SPY), 12.1% (TLT) and 13.8% (GLD),
with a mean difference of 0.03% to 0.08% and a last-day difference of at most 0.055%, so the choice
has no material effect at the forecast date.

**D. Forecast.** The one-day-ahead variance matches (relative difference 0 to machine precision).
Note that this test does not discriminate between conventions, because the initial-variance effect has
decayed away after 4,210 observations.

**E. Simulation (100 repetitions, truth omega 0.02, alpha 0.08, beta 0.90)**

| Parameter | Bias (ours / arch) | RMSE (ours / arch) | Mean absolute difference between estimators | Maximum |
|---|---|---|---|---|
| omega | +0.0050 / +0.0050 | 0.0115 / 0.0115 | 0.00001 | 0.0000 |
| alpha | +0.0002 / +0.0002 | 0.0151 / 0.0151 | 0.00005 | 0.0002 |
| beta | -0.0066 / -0.0066 | 0.0210 / 0.0210 | 0.00001 | 0.0001 |

The disagreement between the two estimators is two to three orders of magnitude below the
sampling error. In no run was our log-likelihood below arch's by more than 0.01 (minimum -0.0004).

## 4. Findings

**Finding 1 (low severity): initial-variance convention differs from `arch`.**
`fit_garch11` starts the recursion at sigma2_0 equal to the sample variance. `arch` treats its
backcast b as the pre-sample squared residual and variance, so its first-day variance is
omega + (alpha + beta) * b. For SPY the two first-day variances are 1.153214 and 1.150245. The
difference decays at the rate (alpha + beta)^t. Evidence: starting our recursion at
omega + (alpha + beta) * b reproduces arch's path to 2.2e-16. Impact: parameters differ by less than
0.0005 and forecasts are unaffected. Recommendation: document the convention in the README. Aligning
to arch's convention is optional and not required for correctness.

**Observation 2 (informational): finite-sample bias is shared.** Both estimators show the same bias
in omega (+0.0050, about 25% of the true 0.02) and beta (-0.0066) at n = 1,500. Because the two
implementations agree to four decimals, this is a property of Gaussian maximum likelihood in
finite samples and not an implementation defect.

**Observation 3 (process): the original test assumption was wrong.** A1 assumed that passing
`backcast=b` makes arch start the variance at b. It does not. The first-run failure was therefore a
test-design error that exposed a real convention difference. The pre-set criteria were kept and the
A1 result is reported as it came out.

## 5. Conclusion

The variance recursion and Gaussian log-likelihood are implemented correctly: with a common
initialization they match the reference to machine precision. The estimator reaches the same
parameters and an equivalent likelihood as `arch` on three real series and in 100 simulated
repetitions. No remediation is required beyond documenting the initialization convention.

## 6. Limitations of this verification

- Only the univariate GARCH(1,1) stage is covered. `arch` has no DCC, so the correlation stage
  (a, b, Q-bar) is not independently re-implemented here. It is checked only by the
  known-correlation recovery in `dcc_garch.py`, which is a weaker test.
- A single reference implementation was used, with Gaussian likelihood and zero mean.
- This is verification (does the code do what is intended). Whether the model is fit for purpose is
  assessed through backtesting and sensitivity analysis, not here.
- Real-data numbers use Yahoo adjusted prices, which are revised over time, so they may shift
  slightly between download dates. The simulation results are reproducible exactly.

## 7. Reproduce

```bash
pip install arch numpy scipy pandas yfinance
python validate_garch_vs_arch.py --end 2026-10-01
```
