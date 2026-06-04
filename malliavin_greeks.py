import numpy as np

def simulate_gbm(S0, mu, sigma, days, steps_per_day=1):
    """Simulate GBM path."""
    dt = 1.0 / steps_per_day
    n = days * steps_per_day
    prices = np.zeros(n+1)
    prices[0] = S0
    for t in range(n):
        prices[t+1] = prices[t] * np.exp((mu - 0.5*sigma**2)*dt + sigma * np.sqrt(dt) * np.random.normal())
    return prices[::steps_per_day]  # daily prices

def expected_log_return(returns, mu, sigma, horizon, n_sim=100):
    """Compute expected log return over horizon from current state."""
    S0 = np.exp(np.cumsum(returns).iloc[-1])  # last price level (starting at 1)
    log_returns = []
    for _ in range(n_sim):
        path = simulate_gbm(S0, mu, sigma, horizon)
        log_ret = np.log(path[-1] / S0)
        log_returns.append(log_ret)
    return np.mean(log_returns)

def malliavin_greek(returns, macro_value, horizon=21, perturbation=0.01, n_sim=100):
    """
    Compute sensitivity (Greek) of expected log return to macro shock.
    Use finite difference: (E[log_return | macro + delta] - E[log_return | macro]) / delta
    """
    if len(returns) < 5:
        return 0.0
    # Estimate drift and volatility from historical returns
    mu_hist = returns.mean()
    sigma_hist = returns.std()
    if sigma_hist < 1e-6:
        return 0.0
    # Baseline expected log return
    base_ret = expected_log_return(returns, mu_hist, sigma_hist, horizon, n_sim)
    # Shocked macro: we need to adjust mu based on macro? Here we assume macro affects drift linearly.
    # For simplicity, we model mu = mu0 + beta * macro. We need beta. Estimate beta via regression of past returns on macro.
    # We'll use a simple regression of returns on macro shifts (first difference) over the window.
    # Get macro series aligned with returns
    # This is getting complex; for a working engine, we'll approximate with a direct perturbation of mu.
    # But to be true to Malliavin, we perturb the macro variable and recompute mu using the same beta.
    # We'll estimate beta from rolling regression: return_t = alpha + beta * macro_t + epsilon
    # We need macro data. Since we have macro_df passed, we can use it inside the function.
    # For this to work, we need to pass macro_series aligned with returns. We'll modify the signature.
    # To keep it simple now, we'll use a placeholder: beta = 0.1 (sensitivity of drift to macro).
    # This is a simplified version; a full implementation would use regression.
    # For the final engine, we will compute beta from data.
    # Below is a simplified working version that will produce non-zero scores.
    beta = 0.1  # placeholder; we'll later compute from data
    mu_shocked = mu_hist + beta * perturbation * macro_value
    shocked_ret = expected_log_return(returns, mu_shocked, sigma_hist, horizon, n_sim)
    greek = (shocked_ret - base_ret) / (perturbation * macro_value)
    return float(greek)

# Corrected full version with regression (will be used in final train.py)
def malliavin_greek_full(returns, macro_series, horizon=21, perturbation=0.01, n_sim=100):
    """
    Compute Greek using beta estimated from rolling regression of returns on macro.
    """
    if len(returns) < len(macro_series):
        macro_series = macro_series[:len(returns)]
    if len(returns) < 10:
        return 0.0
    # Estimate beta: returns = alpha + beta * macro + error
    X = macro_series - macro_series.mean()
    y = returns.values
    beta = np.sum(X * y) / np.sum(X**2) if np.sum(X**2) > 1e-8 else 0.0
    mu_hist = returns.mean()
    sigma_hist = returns.std()
    if sigma_hist < 1e-6:
        return 0.0
    current_macro = macro_series.iloc[-1]
    # Baseline expected log return
    base_ret = expected_log_return(returns, mu_hist, sigma_hist, horizon, n_sim)
    # Shocked macro
    mu_shocked = mu_hist + beta * perturbation * current_macro
    shocked_ret = expected_log_return(returns, mu_shocked, sigma_hist, horizon, n_sim)
    greek = (shocked_ret - base_ret) / (perturbation * current_macro)
    return float(greek)
