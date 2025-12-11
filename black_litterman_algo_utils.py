import numpy as np
import pandas as pd
import math

from scipy.optimize import minimize

from black_litterman_constant import *


# Function to compute returns from NAV
def compute_returns(nav_df):
    """
    Compute simple returns from NAV DataFrame.
    """
    returns = nav_df.pct_change().dropna()
    return returns


# Function to compute covariance matrix from returns
def compute_covariance(returns_df, window=None, method='sample'):
    """
    Compute annualized covariance matrix.
    Assume returns are daily; multiply by 252 for annual.
    If window is specified, use last window rows.
    method: 'sample' or 'ledoit_wolf'
    """
    if window:
        returns_df = returns_df.iloc[-math.ceil(window*weeks_per_year):]

    if method == 'sample':
        cov = returns_df.cov() * weeks_per_year
        return cov

    elif method == 'ledoit_wolf':
        # Implement Ledoit-Wolf shrinkage
        X = returns_df.values  # T x p
        T, p = X.shape

        # Center the data
        mean = np.mean(X, axis=0)
        X = X - mean

        # Sample covariance (biased, as in paper)
        S = np.dot(X.T, X) / T  # p x p

        # m = trace(S) / p
        m = np.trace(S) / p

        # d^2 = ||S - m I||_F^2 / p
        delta_mat = S - m * np.eye(p)
        d2 = np.sum(delta_mat ** 2) / p

        # Compute b^2_tilde = (1/T^2) * sum_t || x_t x_t^T - S ||_F^2 / p
        b2_tilde = 0.0
        for t in range(T):
            outer = np.outer(X[t], X[t]) - S
            b2_tilde += np.sum(outer ** 2) / p
        b2_tilde /= T ** 2

        # b^2 = min(b2_tilde, d2)
        b2 = min(b2_tilde, d2)

        # shrinkage delta* = b^2 / d^2
        shrinkage = 0.0 if d2 == 0 else b2 / d2

        # Shrunk covariance: (1 - shrinkage) * S + shrinkage * m * I
        cov = (1 - shrinkage) * S + shrinkage * m * np.eye(p)

        # Annualize
        cov *= weeks_per_year

        # Return as DataFrame
        return pd.DataFrame(cov, index=returns_df.columns, columns=returns_df.columns)

    else:
        raise ValueError("Unknown method: " + method)

# Black-Litterman model implementation
def black_litterman(
        returns: pd.DataFrame,
        market_caps: pd.Series,
        window: float,
        views: dict,
        # Format: {'absolute': [(asset_sub_or_index, expected_return, confidence)], 'relative': [(asset1_sub_or_index, asset2_sub_or_index, diff, confidence)]}
        tau: float = 0.025,  # Uncertainty in prior
        delta: float = 2.5,  # Risk aversion
        omega_scale: float = 1.0  # For Omega diagonal
):
    """
    Compute Black-Litterman adjusted expected returns.
    Views can use asset sub-classes or index names; internally mapped to indices.
    """
    assets = list(returns.columns)  # indices
    n = len(assets)

    # Equilibrium weights
    w_eq = market_caps / market_caps.sum()
    w_eq_series = pd.Series(w_eq.reindex(assets).fillna(0), index=assets)  # Ensure order matches, fill NaN if any
    w_eq_series.index = [sub_class_map.get(idx, idx) for idx in w_eq_series.index]
    w_eq = w_eq_series.values

    # Covariance
    Sigma = compute_covariance(returns, window)
    Sigma_np = Sigma.values

    # Implied equilibrium returns: mu_eq = delta * Sigma * w_eq
    mu_eq = delta * np.dot(Sigma_np, w_eq)

    # Process views
    k = len(views.get('absolute', [])) + len(views.get('relative', []))
    if k == 0:
        return pd.Series(mu_eq, index=assets), Sigma, w_eq_series  # No views, return equilibrium

    P = np.zeros((k, n))
    Q = np.zeros(k)
    Omega = np.zeros((k, k))

    idx = 0
    asset_idx = {asset: i for i, asset in enumerate(assets)}  # index to position

    def get_index_name(name):
        # Map sub-class to index if needed, else assume it's index
        return reverse_sub_class_map.get(name, name)

    # Absolute views
    for asset, exp_ret, conf in views.get('absolute', []):
        asset_index = get_index_name(asset)
        # if asset_index not in asset_idx:
        #     raise ValueError(f"Unknown asset: {asset} (mapped to {asset_index})")
        if asset_index in asset_idx:
            i = asset_idx[asset_index]
            P[idx, i] = 1
            Q[idx] = exp_ret
            Omega[idx, idx] = (1 - conf) / conf * omega_scale if conf > 0 else 1e6  # Uncertainty; conf in (0,1]
            idx += 1

    # Relative views: asset1 - asset2 = diff
    for asset1, asset2, diff, conf in views.get('relative', []):
        asset1_index = get_index_name(asset1)
        asset2_index = get_index_name(asset2)
        # if asset1_index not in asset_idx or asset2_index not in asset_idx:
        #     raise ValueError(f"Unknown assets: {asset1} ({asset1_index}), {asset2} ({asset2_index})")
        if asset1_index in asset_idx and asset2_index in asset_idx:
            i1, i2 = asset_idx[asset1_index], asset_idx[asset2_index]
            P[idx, i1] = 1
            P[idx, i2] = -1
            Q[idx] = diff
            Omega[idx, idx] = (1 - conf) / conf * omega_scale if conf > 0 else 1e6
            idx += 1

    # BL formula
    tau_Sigma = tau * Sigma_np
    try:
        M = np.linalg.inv(np.dot(P, np.dot(tau_Sigma, P.T)) + Omega)
    except np.linalg.LinAlgError:
        # If singular, add small jitter
        M = np.linalg.inv(np.dot(P, np.dot(tau_Sigma, P.T)) + Omega + np.eye(k) * 1e-6)
    adjustment = np.dot(tau_Sigma, np.dot(P.T, np.dot(M, (Q - np.dot(P, mu_eq)))))
    mu_bl = mu_eq + adjustment
    mu_series = pd.Series(mu_bl, index=assets)
    mu_series.index = [sub_class_map.get(idx, idx) for idx in mu_series.index]


    # Posterior covariance (He-Litterman approximation)
    Sigma_bl = Sigma_np + tau_Sigma - np.dot(tau_Sigma, np.dot(P.T, np.dot(M, np.dot(P, tau_Sigma))))

    return mu_series, pd.DataFrame(Sigma_bl, index=assets, columns=assets), w_eq_series


def rescale_equilibrium_weights(w_eq_0: pd.Series, risk_level: str) -> pd.Series:
    """
    Rescale market equilibrium weights to match target broad asset class allocation
    based on risk level, while preserving internal (sub-class) ratios.

    Parameters:
    - w_eq_0: original equilibrium weights (sub-class indexed)
    - risk_level: 'C1', 'C2', 'C3', 'C4', or 'C5'

    Returns:
    - w_eq_rescaled: rescaled weights (same index as w_eq_0)
    """
    # 1. Compute broad class totals from w_eq_0
    broad_totals = {}
    for sub, w in w_eq_0.items():
        broad = sub_class_to_broad_class.get(sub)
        if broad:
            broad_totals[broad] = broad_totals.get(broad, 0.0) + w

    # Handle missing broad classes
    for broad in ['权益', '债券', '货币', '大宗商品', '另类投资']:
        if broad not in broad_totals:
            broad_totals[broad] = 0.0

    # 2. Define target broad class weights
    if risk_level == 'C1':
        target_bonds = 0.60
        target_equity = None  # not controlled
        target_others = 1.0 - target_bonds
        broad_order = ['债券']
    elif risk_level == 'C2':
        target_bonds = 0.40
        target_equity = None
        target_others = 1.0 - target_bonds
        broad_order = ['债券']
    elif risk_level in ['C3', 'C4', 'C5']:
        equity_map = {'C3': 0.50, 'C4': 0.60, 'C5': 0.70}
        target_equity = equity_map[risk_level]
        target_bonds = None
        target_others = 1.0 - target_equity
        broad_order = ['权益']
    else:
        raise ValueError(f"Unsupported risk_level: {risk_level}")

    # 3. Identify "other" broad classes (not controlled)
    controlled_broad = broad_order[0]
    other_broads = [b for b in ['权益', '债券', '货币', '大宗商品', '另类投资'] if b != controlled_broad]

    # 4. Compute scaling factors
    scaling_factors = {}

    # Controlled broad class
    current_controlled = broad_totals[controlled_broad]
    if current_controlled > 1e-8:
        scaling_factors[controlled_broad] = (target_equity if target_equity is not None else target_bonds) / current_controlled
    else:
        scaling_factors[controlled_broad] = 0.0

    # Other broad classes: scale proportionally to their current share of "others"
    current_others_total = sum(broad_totals[b] for b in other_broads)
    if current_others_total > 1e-8:
        for broad in other_broads:
            current_share = broad_totals[broad] / current_others_total
            scaling_factors[broad] = (target_others * current_share) / broad_totals[broad] if broad_totals[broad] > 1e-8 else 0.0
    else:
        # If no "other" assets, distribute evenly (shouldn't happen)
        for broad in other_broads:
            scaling_factors[broad] = target_others / len(other_broads) if broad_totals[broad] > 1e-8 else 0.0

    # 5. Apply scaling to each sub-class
    w_eq_rescaled = pd.Series(0.0, index=w_eq_0.index)
    for sub, w in w_eq_0.items():
        broad = sub_class_to_broad_class.get(sub)
        if broad and broad in scaling_factors:
            w_eq_rescaled[sub] = w * scaling_factors[broad]
        else:
            w_eq_rescaled[sub] = w  # fallback

    # Normalize to sum to 1.0 (handle rounding errors)
    total = w_eq_rescaled.sum()
    if total > 1e-8:
        w_eq_rescaled = w_eq_rescaled / total

    return w_eq_rescaled


def get_portfolio_weight(input_param: dict = None):
    optimization_method = input_param.get('optimization_method', 'equal_weighted')
    risk_level = input_param.get('risk_level', 'C1')
    mu = input_param.get('expected_return')
    sigma = input_param.get('covariance_matrix')

    constraints = {
        'min_weight': 0.005,
        'max_weight': 0.2,
    }
    if risk_level in ['C3', 'C4', 'C5']:
        equity_indices = [i for i, asset in enumerate(mu.index) if sub_class_to_broad_class.get(asset) == '权益']
        equity_min = 0.1 + 0.1 * int(risk_level[-1])
        equity_max = 0.3 + 0.1 * int(risk_level[-1])
        constraints['groups'] = [{'indices': equity_indices, 'min': equity_min, 'max': equity_max}]

    if optimization_method == 'mean_variance':
        target_vol = input_param.get('target_vol', 0)
        w_eq = input_param.get('benchmark_weights', None)
        benchmark_ratios = {}
        tol = input_param.get('tol', 0.2)

        if w_eq is not None:
            benchmark_ratios, _ = compute_benchmark_ratios(w_eq)

        weights, vol = mean_variance_optimization(
            mu, sigma, target_vol=target_vol,
            constraints=constraints,
            benchmark_ratios=benchmark_ratios,
            tol=tol,
            sub_class_to_broad_class=sub_class_to_broad_class
        )
    elif optimization_method == 'risk_parity':
        weights, var = risk_parity_optimization(sigma, constraints=constraints)
        vol = np.sqrt(var)
    else:
        weights = pd.Series(index=mu.index, data=1 / len(mu))
        vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))

    weights.index = [sub_class_map.get(idx, idx) for idx in weights.index]
    return weights, vol


def compute_benchmark_ratios(w_bench: pd.Series, broad_classes=('权益', '债券')):
    ratios = {}
    # First, get broad class totals from benchmark
    broad_totals = {}
    for sub, w in w_bench.items():
        if sub in sub_class_to_broad_class:
            broad = sub_class_to_broad_class[sub]
            if broad in broad_classes:
                broad_totals[broad] = broad_totals.get(broad, 0.0) + w

    # Then compute ratios
    for sub, w in w_bench.items():
        if sub in sub_class_to_broad_class:
            broad = sub_class_to_broad_class[sub]
            if broad in broad_classes and broad_totals[broad] > 1e-8:
                ratios[sub] = w / broad_totals[broad]
    return ratios, broad_totals


def mean_variance_optimization(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    target_vol: float,
    constraints: dict = None,
    benchmark_ratios: dict = None,   # e.g., {'美股': 0.4, '新兴市场': 0.3, ...}
    tol: float = 0.2,                # ±20% tolerance
    sub_class_to_broad_class: dict = None
):
    if constraints is None:
        constraints = {}
    if benchmark_ratios is None:
        benchmark_ratios = {}
    if sub_class_to_broad_class is None:
        sub_class_to_broad_class = {}

    n = len(mu)
    mu_np = mu.values
    Sigma_np = Sigma.values
    target_var = target_vol ** 2
    asset_names = mu.index.tolist()

    # Bounds and sum-to-one
    min_w = constraints.get('min_weight', 0.0)
    max_w = constraints.get('max_weight', 1.0)
    bounds = [(min_w, max_w) for _ in range(n)]
    sum_cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    # Group (absolute) constraints
    group_cons = []
    if 'groups' in constraints:
        for group in constraints['groups']:
            idx = group['indices']
            min_g = group.get('min', min_w)
            max_g = group.get('max', max_w)
            group_cons.append({'type': 'ineq', 'fun': lambda w, idx=idx, min_g=min_g: np.sum(w[idx]) - min_g})
            group_cons.append({'type': 'ineq', 'fun': lambda w, idx=idx, max_g=max_g: max_g - np.sum(w[idx])})

    # === NEW: Nonlinear within-broad-class ratio constraints ===
    nonlinear_cons = []

    # Group sub-indices by broad class
    broad_to_indices = {}
    for i, sub in enumerate(asset_names):
        broad = sub_class_to_broad_class.get(sub, 'Other')
        if broad in ['权益', '债券'] and sub in benchmark_ratios:
            if broad not in broad_to_indices:
                broad_to_indices[broad] = []
            broad_to_indices[broad].append(i)

    # For each broad class, define constraints
    for broad, indices in broad_to_indices.items():
        # Build a mask for this broad class
        mask = np.zeros(n, dtype=bool)
        mask[indices] = True

        # For each sub-class in this broad class
        for i in indices:
            sub = asset_names[i]
            r_bench = benchmark_ratios[sub]
            r_low = r_bench - tol
            r_high = r_bench + tol

            # Constraint 1: w_i >= r_low * sum_{j in broad} w_j
            # → w_i - r_low * sum(w_j for j in broad) >= 0
            def cons_low(w, i=i, mask=mask, r_low=r_low):
                return w[i] - r_low * np.sum(w[mask])

            # Constraint 2: w_i <= r_high * sum_{j in broad} w_j
            # → r_high * sum(w_j for j in broad) - w_i >= 0
            def cons_high(w, i=i, mask=mask, r_high=r_high):
                return r_high * np.sum(w[mask]) - w[i]

            nonlinear_cons.append({'type': 'ineq', 'fun': cons_low})
            nonlinear_cons.append({'type': 'ineq', 'fun': cons_high})

    # Min variance feasibility check
    def min_var_obj(w):
        return np.dot(w.T, np.dot(Sigma_np, w))

    all_constraints = [sum_cons] + group_cons + nonlinear_cons
    res_min_var = minimize(
        min_var_obj, x0=np.ones(n)/n,
        method='SLSQP',
        bounds=bounds,
        constraints=all_constraints,
        options={'maxiter': 1000}
    )

    if not res_min_var.success:
        raise ValueError("Min var failed: " + str(res_min_var.message))

    min_var = res_min_var.fun
    if min_var > target_var:
        return pd.Series(res_min_var.x, index=mu.index), np.sqrt(min_var)

    # Max return optimization
    def max_ret_obj(w):
        return -np.dot(mu_np, w)

    var_cons = {'type': 'ineq', 'fun': lambda w: target_var - np.dot(w.T, np.dot(Sigma_np, w))}
    all_constraints = [sum_cons, var_cons] + group_cons + nonlinear_cons

    res = minimize(
        max_ret_obj, x0=res_min_var.x,
        method='SLSQP',
        bounds=bounds,
        constraints=all_constraints,
        options={'maxiter': 1000}
    )

    if res.success:
        final_vol = np.sqrt(np.dot(res.x.T, np.dot(Sigma_np, res.x)))
        return pd.Series(res.x, index=mu.index), final_vol
    else:
        raise ValueError("Max return failed: " + str(res.message))


# Risk Parity Optimization
def risk_parity_optimization(
        Sigma: pd.DataFrame,
        b: np.array = None,  # Risk budget, default equal
        constraints: dict = None
):
    """
    Risk parity: minimize sum (RC_i - b_i * sigma_p)^2 where RC_i = w_i * (Sigma w)_i / sigma_p
    """
    if constraints is None:
        constraints = {}
    n = Sigma.shape[0]
    Sigma_np = Sigma.values

    if b is None:
        b = np.ones(n) / n

    def objective(w):
        portfolio_variance = np.dot(w.T, np.dot(Sigma_np, w))
        sigma_p = np.sqrt(portfolio_variance)
        marginal_risks = np.dot(Sigma_np, w)
        risk_contribs = w * marginal_risks / sigma_p
        return np.sum((risk_contribs - b * sigma_p) ** 2)

    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    min_w = constraints.get('min_weight', 0.0)
    max_w = constraints.get('max_weight', 1.0)
    bounds = [(min_w, max_w) for _ in range(n)]

    # Add group constraints if any
    if 'groups' in constraints:
        for group in constraints['groups']:
            idx = group['indices']
            min_g = group.get('min', min_w)
            max_g = group.get('max', max_w)
            cons.append({'type': 'ineq', 'fun': lambda w, idx=idx, min_g=min_g: np.sum(w[idx]) - min_g})
            cons.append({'type': 'ineq', 'fun': lambda w, idx=idx, max_g=max_g: max_g - np.sum(w[idx])})

    # Tracking error constraint
    if 'benchmark_weights' in constraints and 'max_tracking_error' in constraints:
        b_w = constraints['benchmark_weights'].values
        max_te_var = constraints['max_tracking_error'] ** 2
        cons.append({'type': 'ineq', 'fun': lambda w: max_te_var - np.dot((w - b_w).T, np.dot(Sigma_np, (w - b_w)))})

    x0 = np.ones(n) / n
    res = minimize(objective, x0=x0, method='SLSQP', bounds=bounds, constraints=cons)

    if res.success:
        return pd.Series(res.x, index=Sigma.index), np.dot(res.x.T, np.dot(Sigma_np, res.x))
    else:
        raise ValueError("Optimization failed: " + res.message)

def backtest_portfolio(weights: pd.Series, returns_df: pd.DataFrame, window: float, rebalance: bool = False):
    """
    Backtest portfolio performance over the last `window` of returns data.

    Parameters:
    - weights: Portfolio weights (pandas Series), indexed by asset sub-class names.
    - returns_df: Full historical returns DataFrame (columns = index names like 'SCHG').
    - window: Backtesting period in years (e.g., 1.5).
    - rebalance: If True, rebalance to target weights every period (weekly).
                 If False (default), buy-and-hold (no rebalancing).

    Returns:
    - nav: Cumulative NAV (starting at 1.0)
    - stats: Dict with performance metrics
    """
    n_weeks = int(window * weeks_per_year)

    # Map sub-class names back to original index names
    sub_class_to_index = {v: k for k, v in sub_class_map.items()}
    weight_original_index = {}
    for sub_class, w in weights.items():
        if sub_class in sub_class_to_index:
            idx_name = sub_class_to_index[sub_class]
            weight_original_index[idx_name] = w
        else:
            weight_original_index[sub_class] = w

    # Align with returns_df columns
    aligned_weights = pd.Series(0.0, index=returns_df.columns)
    for idx in aligned_weights.index:
        if idx in weight_original_index:
            aligned_weights[idx] = weight_original_index[idx]

    # Trim to backtest window
    returns_window = returns_df.iloc[-n_weeks:]

    if rebalance:
        # Rebalanced: same as before
        port_returns = returns_window.dot(aligned_weights)
        nav = (1 + port_returns).cumprod()
        nav = pd.concat([pd.Series([1.0], index=[returns_window.index[0] - pd.Timedelta(days=7)]), nav])
    else:
        # Buy-and-hold: simulate NAV path per asset
        # Start with $1 total portfolio
        initial_value_per_asset = aligned_weights  # since total = 1
        # Compute cumulative NAV for each asset from returns
        asset_nav = (1 + returns_window).cumprod()
        # Scale by initial allocation
        portfolio_nav_components = asset_nav.mul(initial_value_per_asset, axis=1)
        # Sum across assets
        nav = portfolio_nav_components.sum(axis=1)
        # Prepend starting point (1.0) at time t=0
        start_date = returns_window.index[0] - pd.Timedelta(days=7)
        nav = pd.concat([pd.Series([1.0], index=[start_date]), nav])

    # Performance stats (annualized, based on actual weekly returns)
    port_returns_actual = nav.pct_change().dropna()
    total_weeks = len(port_returns_actual)
    if total_weeks == 0:
        stats = {'Annualized Return': 0, 'Annualized Volatility': 0, 'Sharpe Ratio': 0, 'Max Drawdown': 0}
    else:
        annualized_return = (nav.iloc[-1] / nav.iloc[0]) ** (weeks_per_year / total_weeks) - 1
        annualized_vol = port_returns_actual.std() * np.sqrt(weeks_per_year)
        sharpe = annualized_return / annualized_vol if annualized_vol != 0 else 0
        drawdown = (nav / nav.cummax() - 1)
        max_dd = drawdown.min()

        stats = {
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd
        }

    return nav, stats


def get_rebalance_dates(
        returns_df: pd.DataFrame,
        period_years: int,
        freq: str = 'Q'  # 'Q', '2Q', 'Y', etc.
) -> pd.DatetimeIndex:
    """
    Generate rebalance dates aligned to the index of returns_df.
    Uses the last available date <= each theoretical rebalance date.
    """
    end_date = returns_df.index[-1]
    start_date = end_date - pd.DateOffset(years=period_years)

    # Generate theoretical rebalance dates (e.g., quarter-ends)
    theoretical_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq=freq
    )

    # Find the latest index date <= each theoretical date
    aligned_indices = returns_df.index.get_indexer(theoretical_dates, method='pad')

    # Filter out -1 (dates before start of data)
    valid_indices = aligned_indices[aligned_indices >= 0]
    aligned_dates = returns_df.index[valid_indices]

    # Remove duplicates and sort
    aligned_dates = aligned_dates.unique().sort_values()

    if len(aligned_dates) == 0:
        raise ValueError("No valid rebalance dates found. Check data range and frequency.")

    return aligned_dates


def find_last_index_on_or_before(idx: pd.DatetimeIndex, target_date: pd.Timestamp) -> int:
    """
    Find the integer position of the last index <= target_date.
    Raises ValueError if no such index exists.
    """
    pos = idx.get_indexer([target_date], method='pad')
    if pos[0] == -1:
        raise ValueError(f"No data on or before {target_date}")
    return int(pos)


def rolling_rebalance_backtest(
        returns_df: pd.DataFrame,
        market_caps: pd.Series,
        views: dict,
        period_years: int,
        rebalance_freq: str,
        optimization_method: str,
        target_vol: float = None,
        window: float = 1.5,  # years of historical data used for BL at each rebalance
        risk_level: str = 'C3',
        tol: float = 0.20  # tolerance for within-broad-class ratio constraints
):
    """
    Perform a rolling out-of-sample backtest with periodic rebalancing.

    Parameters:
    - returns_df: Weekly returns DataFrame (index = dates, columns = index names like 'SCHG')
    - market_caps: Static market cap weights (pd.Series, index = index names)
    - views: Black-Litterman views dict
    - period_years: Length of backtest period (e.g., 5 for last 5 years)
    - rebalance_freq: Rebalance frequency ('Q' = quarterly, '2Q' = semi-annual, 'Y' = annual)
    - optimization_method: 'mean_variance' or 'risk_parity'
    - target_vol: Target volatility (only for mean_variance)
    - window: Years of data to use for covariance/BL at each rebalance
    - risk_level: e.g., 'C3'
    - tol: Tolerance for internal allocation drift (only for mean_variance)

    Returns:
    - full_nav: pd.Series of cumulative NAV (base = 1.0)
    - portfolio_history: dict {rebalance_date: weights (sub-class indexed)}
    - stats: dict of performance metrics
    """
    # 1. Generate rebalance dates aligned to data
    rebalance_dates = get_rebalance_dates(returns_df, period_years, rebalance_freq)
    if len(rebalance_dates) < 1:
        raise ValueError("No valid rebalance dates found.")
    print(f"Found {len(rebalance_dates)} rebalance dates from {rebalance_dates[0].date()} to {rebalance_dates[-1].date()}")

    # 2. Initialize
    full_nav = pd.Series([1.0], index=[returns_df.index[0]])
    full_nav = pd.Series([1.0], index=[rebalance_dates[0]])
    portfolio_history = {}

    # 3. Loop over each rebalance event
    for i, reb_date in enumerate(rebalance_dates):
        try:
            # Get position of reb_date in returns index
            current_pos = find_last_index_on_or_before(returns_df.index, reb_date)

            # Build historical window for BL: [current_pos - lookback + 1, current_pos]
            lookback_weeks = int(window * weeks_per_year)
            window_start_idx = current_pos - lookback_weeks + 1
            if window_start_idx < 0:
                window_start_idx = 0
            bl_returns = returns_df.iloc[window_start_idx: current_pos + 1]

            # Skip if too little data
            if len(bl_returns) < 10:  # require at least 10 weeks
                print(f"  ⚠️ Skipping {reb_date.date()}: insufficient data ({len(bl_returns)} weeks)")
                continue

            # Align market caps to available assets in bl_returns
            available_assets = bl_returns.columns
            market_caps_aligned = market_caps[market_caps.index.isin(available_assets)]
            if market_caps_aligned.empty:
                print(f"  ⚠️ Skipping {reb_date.date()}: no market cap data")
                continue

            # Run Black-Litterman
            mu_bl, Sigma_bl, w_eq = black_litterman(
                returns=bl_returns,
                market_caps=market_caps_aligned,
                window=window,
                views=views,
                tau=0.025,
                delta=2.5
            )

            # Prepare optimization parameters
            opt_params = {
                'risk_level': risk_level,
                'expected_return': mu_bl,
                'covariance_matrix': Sigma_bl,
                'optimization_method': optimization_method,
                'benchmark_weights': w_eq
            }
            if optimization_method == 'mean_variance':
                opt_params['target_vol'] = target_vol
                # Compute benchmark ratios for internal allocation constraints
                broad_totals = {}
                for sub, w in w_eq.items():
                    broad = sub_class_to_broad_class.get(sub)
                    if broad in ['权益', '债券']:
                        broad_totals[broad] = broad_totals.get(broad, 0.0) + w
                benchmark_ratios = {}
                for sub, w in w_eq.items():
                    broad = sub_class_to_broad_class.get(sub)
                    if broad in broad_totals and broad_totals[broad] > 1e-8:
                        benchmark_ratios[sub] = w / broad_totals[broad]
                opt_params['benchmark_ratios'] = benchmark_ratios
                opt_params['tol'] = tol

            # Optimize
            weights, vol = get_portfolio_weight(opt_params)
            portfolio_history[reb_date] = weights.copy()

            # Determine holding period end
            if i == len(rebalance_dates) - 1:
                hold_end_date = returns_df.index[-1]
            else:
                # Hold until the day BEFORE next rebalance
                next_reb_date = rebalance_dates[i + 1]
                hold_end_date = next_reb_date - pd.Timedelta(days=1)

            # Find data boundaries for holding period
            hold_start_idx = current_pos + 1
            try:
                hold_end_idx = find_last_index_on_or_before(returns_df.index, hold_end_date)
            except ValueError:
                hold_end_idx = len(returns_df) - 1

            if hold_start_idx > hold_end_idx:
                continue

            hold_returns = returns_df.iloc[hold_start_idx: hold_end_idx + 1]
            if hold_returns.empty:
                continue

            # Map weights to original index names for dot product
            sub_class_to_index = {v: k for k, v in sub_class_map.items()}
            aligned_weights = pd.Series(0.0, index=hold_returns.columns)
            for sub_class, w in weights.items():
                idx_name = sub_class_to_index.get(sub_class, sub_class)
                if idx_name in aligned_weights.index:
                    aligned_weights[idx_name] = w

            # Compute portfolio returns (buy-and-hold during holding period)
            port_returns = hold_returns.dot(aligned_weights)
            if port_returns.empty:
                continue

            # Chain to full NAV
            gross_returns = 1 + port_returns
            segment_growth = gross_returns.cumprod()
            segment_nav = pd.Series(
                full_nav.iloc[-1] * segment_growth.values,
                index=port_returns.index
            )
            full_nav = pd.concat([full_nav, segment_nav])

        except Exception as e:
            print(f"  ⚠️ Failed at rebalance {reb_date.date()}: {str(e)}")
            continue

    # Clean NAV series
    full_nav = full_nav[~full_nav.index.duplicated(keep='first')].sort_index()

    # Compute performance stats
    if len(full_nav) < 2:
        stats = {'Annualized Return': 0, 'Annualized Volatility': 0, 'Sharpe Ratio': 0, 'Max Drawdown': 0}
    else:
        port_returns_actual = full_nav.pct_change().dropna()
        total_weeks = len(port_returns_actual)
        annualized_return = (full_nav.iloc[-1] / full_nav.iloc[0]) ** (weeks_per_year / total_weeks) - 1
        annualized_vol = port_returns_actual.std() * np.sqrt(weeks_per_year)
        sharpe = annualized_return / annualized_vol if annualized_vol != 0 else 0
        drawdown = (full_nav / full_nav.cummax() - 1)
        max_dd = drawdown.min()
        stats = {
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd
        }

    return full_nav, portfolio_history, stats


# Helper to Remove One View
def remove_view_at_index(views: dict, view_type: str, index: int) -> dict:
    """
    Remove one view from the views dict.
    Returns a new dict without modifying the original.
    """
    new_views = {
        'absolute': views.get('absolute', []).copy(),
        'relative': views.get('relative', []).copy()
    }
    if view_type == 'absolute' and index < len(new_views['absolute']):
        del new_views['absolute'][index]
    elif view_type == 'relative' and index < len(new_views['relative']):
        del new_views['relative'][index]
    return new_views

# View Sensitivity Analysis
def analyze_view_sensitivity(
        returns_df: pd.DataFrame,
        market_caps: pd.Series,
        full_views: dict,
        window: float,
        optimization_method: str,
        risk_level: str = 'C3',
        target_vol: float = None,
        tol: float = 0.20
):
    """
    Perform view sensitivity analysis.

    Returns:
      - baseline_weights: weights with all views
      - sensitivity_results: list of {
            'view_desc': str,
            'weights': pd.Series,
            'total_abs_change': float,
            'top_changes': pd.Series  # top 5 weight changes
        }
    """
    # 1. Compute baseline (full views)
    print("Computing baseline portfolio (all views)...")
    mu_full, Sigma_full, w_eq_full = black_litterman(returns_df, market_caps, window, full_views)

    opt_params_full = {
        'risk_level': risk_level,
        'expected_return': mu_full,
        'covariance_matrix': Sigma_full,
        'optimization_method': optimization_method,
        'benchmark_weights': w_eq_full
    }
    if optimization_method == 'mean_variance':
        opt_params_full['target_vol'] = target_vol
        # Add benchmark ratios (as before)
        broad_totals = {}
        for sub, w in w_eq_full.items():
            broad = sub_class_to_broad_class.get(sub)
            if broad in ['权益', '债券']:
                broad_totals[broad] = broad_totals.get(broad, 0.0) + w
        benchmark_ratios = {}
        for sub, w in w_eq_full.items():
            broad = sub_class_to_broad_class.get(sub)
            if broad in broad_totals and broad_totals[broad] > 1e-8:
                benchmark_ratios[sub] = w / broad_totals[broad]
        opt_params_full['benchmark_ratios'] = benchmark_ratios
        opt_params_full['tol'] = tol

    baseline_weights, _ = get_portfolio_weight(opt_params_full)
    # Backtest baseline portfolio
    nav_baseline, stats_baseline = backtest_portfolio(
        baseline_weights, returns_df, window=window, rebalance=False
    )

    # 2. Generate all "leave-one-out" view sets
    all_view_configs = []

    # Absolute views
    for i, view in enumerate(full_views.get('absolute', [])):
        view_desc = f"Absolute: {view[0]} → {view[1]:.1%} (conf={view[2]:.0%})"
        reduced_views = remove_view_at_index(full_views, 'absolute', i)
        all_view_configs.append((view_desc, reduced_views))

    # Relative views
    for i, view in enumerate(full_views.get('relative', [])):
        view_desc = f"Relative: {view[0]} - {view[1]} → {view[2]:+.1%} (conf={view[3]:.0%})"
        reduced_views = remove_view_at_index(full_views, 'relative', i)
        all_view_configs.append((view_desc, reduced_views))

    if not all_view_configs:
        print("No views to analyze.")
        return baseline_weights, []

    # 3. Compute weights for each reduced view set
    sensitivity_results = []
    print(f"\nAnalyzing {len(all_view_configs)} leave-one-out portfolios...")

    for view_desc, views in all_view_configs:
        try:
            mu, Sigma, w_eq = black_litterman(returns_df, market_caps, window, views)
            opt_params = {
                'risk_level': risk_level,
                'expected_return': mu,
                'covariance_matrix': Sigma,
                'optimization_method': optimization_method,
                'benchmark_weights': w_eq
            }
            if optimization_method == 'mean_variance':
                opt_params['target_vol'] = target_vol
                # Recompute benchmark ratios from w_eq
                broad_totals = {}
                for sub, w in w_eq.items():
                    broad = sub_class_to_broad_class.get(sub)
                    if broad in ['权益', '债券']:
                        broad_totals[broad] = broad_totals.get(broad, 0.0) + w
                benchmark_ratios = {}
                for sub, w in w_eq.items():
                    broad = sub_class_to_broad_class.get(sub)
                    if broad in broad_totals and broad_totals[broad] > 1e-8:
                        benchmark_ratios[sub] = w / broad_totals[broad]
                opt_params['benchmark_ratios'] = benchmark_ratios
                opt_params['tol'] = tol

            weights, _ = get_portfolio_weight(opt_params)

            # Backtest this portfolio
            nav_variant, stats_variant = backtest_portfolio(
                weights, returns_df, window=window, rebalance=False
            )

            # Compute weight change (WITH - WITHOUT)
            all_assets = baseline_weights.index.union(weights.index)
            base_aligned = baseline_weights.reindex(all_assets, fill_value=0.0)
            weights_aligned = weights.reindex(all_assets, fill_value=0.0)
            view_impact = base_aligned - weights_aligned  # positive = view increased weight
            total_impact = view_impact.abs().sum()

            # Store backtest stats
            perf_impact = {
                'Return': stats_baseline['Annualized Return'] - stats_variant['Annualized Return'],
                'Volatility': stats_baseline['Annualized Volatility'] - stats_variant['Annualized Volatility'],
                'Sharpe': stats_baseline['Sharpe Ratio'] - stats_variant['Sharpe Ratio'],
                'MaxDD': stats_baseline['Max Drawdown'] - stats_variant['Max Drawdown']
            }

            sensitivity_results.append({
                'view_desc': view_desc,
                'weights': weights_aligned,
                'view_impact': view_impact,
                'total_weight_impact': total_impact,
                'backtest_stats': stats_variant,
                'perf_impact': perf_impact,  # baseline - variant = contribution of view
                'nav': nav_variant  # optional: store NAV for plotting
            })

        except Exception as e:
            print(f"  ✗ Failed for {view_desc[:50]}...: {str(e)}")
            continue

    # 4. Sort by impact (total absolute weight change)
    sensitivity_results.sort(key=lambda x: x['total_weight_impact'], reverse=True)

    return baseline_weights, sensitivity_results




















