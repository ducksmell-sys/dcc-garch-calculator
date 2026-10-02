"""VaR backtest statistics: Kupiec (coverage) and Christoffersen (independence) tests.

Same tests as the standalone var-backtester repo, included here so this repo is self-contained.
"""

import numpy as np
from scipy.stats import chi2


def _binom_loglik(prob, successes, total):
    if successes == 0:
        return (total - successes) * np.log(1 - prob)
    if successes == total:
        return successes * np.log(prob)
    return (total - successes) * np.log(1 - prob) + successes * np.log(prob)


def kupiec_test(violations, confidence=0.95):
    """Is the observed violation rate consistent with 1 - confidence?"""
    violations = np.asarray(violations)
    n, x = len(violations), int(violations.sum())
    p = 1 - confidence
    lr = -2 * (_binom_loglik(p, x, n) - _binom_loglik(x / n, x, n))
    return {"statistic": lr, "p_value": 1 - chi2.cdf(lr, df=1), "violations": x, "total": n, "rate": x / n}


def christoffersen_independence_test(violations):
    """Are violations scattered randomly, or clustered in time?"""
    v = np.asarray(violations).astype(int)
    n00 = n01 = n10 = n11 = 0
    for prev, curr in zip(v[:-1], v[1:]):
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def lg(x):
        return np.log(x) if x > 0 else 0.0

    ll_restricted = (n00 + n10) * lg(1 - pi) + (n01 + n11) * lg(pi)
    ll_unrestricted = n00 * lg(1 - pi01) + n01 * lg(pi01) + n10 * lg(1 - pi11) + n11 * lg(pi11)
    lr = -2 * (ll_restricted - ll_unrestricted)
    return {"statistic": lr, "p_value": 1 - chi2.cdf(lr, df=1)}


def conditional_coverage_test(violations, confidence=0.95):
    """Kupiec + independence, chi-squared with 2 degrees of freedom."""
    lr = kupiec_test(violations, confidence)["statistic"] + christoffersen_independence_test(violations)["statistic"]
    return {"statistic": lr, "p_value": 1 - chi2.cdf(lr, df=2)}
