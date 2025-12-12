import numpy as np
import pandas as pd
import math

import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize

from black_litterman_constant import *
from black_litterman_algo_utils import *

try:
    from wcwidth import wcswidth
except ImportError:
    # Fallback: treat all chars as width 1 (less accurate but works)
    def wcswidth(s):
        return len(s)


def print_internal_allocation(
        weights_bench: pd.Series,
        weights_opt: pd.Series,
        title: str = "Portfolio"
):
    """
    Print internal allocation ratios with proper alignment for mixed Chinese/English.
    """
    broad_classes_of_interest = ['权益', '债券']

    def get_broad_totals(weights):
        totals = {broad: 0.0 for broad in broad_classes_of_interest}
        for sub, w in weights.items():
            broad = sub_class_to_broad_class.get(sub)
            if broad in broad_classes_of_interest:
                totals[broad] += w
        return totals

    bench_totals = get_broad_totals(weights_bench)
    opt_totals = get_broad_totals(weights_opt)

    all_subs = set()
    for weights in [weights_bench, weights_opt]:
        for sub in weights.index:
            if sub in sub_class_to_broad_class and sub_class_to_broad_class[sub] in broad_classes_of_interest:
                all_subs.add(sub)
    all_subs = sorted(all_subs)

    # --- Use wcwidth for alignment ---
    try:
        from wcwidth import wcswidth
    except ImportError:
        def wcswidth(s):
            return sum(2 if '\u4e00' <= c <= '\u9fff' else 1 for c in s)

    # Header
    header1 = "Sub-Class"
    header2 = "Broad"
    header3 = "Benchmark Ratio"
    header4 = "Optimized Ratio"
    header5 = "Abs Diff"

    # Compute display widths
    col1_width = max(wcswidth(sub) for sub in all_subs + [header1]) + 2
    col2_width = max(wcswidth(broad) for broad in broad_classes_of_interest + [header2]) + 2
    col3_width = len(header3) + 2
    col4_width = len(header4) + 2
    col5_width = len(header5)

    # Build header line
    def pad_to_width(text, target_width):
        current = wcswidth(text)
        if current < target_width:
            return text + ' ' * (target_width - current)
        return text

    line1 = (
            pad_to_width(header1, col1_width) +
            pad_to_width(header2, col2_width) +
            pad_to_width(header3, col3_width) +
            pad_to_width(header4, col4_width) +
            pad_to_width(header5, col5_width)
    )
    print(f"\n=== Internal Allocation in (Benchmark vs {title}) ===")
    print(line1)
    print("-" * (col1_width + col2_width + col3_width + col4_width + col5_width))

    # Data rows
    for sub in all_subs:
        broad = sub_class_to_broad_class.get(sub, "Unknown")
        if broad not in broad_classes_of_interest:
            continue

        w_bench = weights_bench.get(sub, 0.0)
        total_bench = bench_totals[broad]
        ratio_bench = w_bench / total_bench if total_bench > 1e-8 else 0.0

        w_opt = weights_opt.get(sub, 0.0)
        total_opt = opt_totals[broad]
        ratio_opt = w_opt / total_opt if total_opt > 1e-8 else 0.0

        diff = abs(ratio_opt - ratio_bench)

        row = (
                pad_to_width(sub, col1_width) +
                pad_to_width(broad, col2_width) +
                pad_to_width(f"{ratio_bench:>12.2%}", col3_width) +
                pad_to_width(f"{ratio_opt:>12.2%}", col4_width) +
                pad_to_width(f"{diff:>9.2%}", col5_width)
        )
        print(row)

    # Broad class totals
    print("\nBroad Class Totals:")
    for broad in broad_classes_of_interest:
        total_b = bench_totals[broad]
        total_o = opt_totals[broad]
        line = f"  {broad:<8} → Benchmark: {total_b:>7.2%}, Optimized: {total_o:>7.2%}"
        print(line)

def adjust_splits_in_nav(nav_df, etf_suffix='US Equity', max_daily_change=0.25):
    """
    Adjusts ETF price series in nav_df for splits by detecting large jumps
    and applying backward adjustment factors.

    Parameters:
    - nav_df: DataFrame with datetime index and ticker columns
    - etf_suffix: only adjust columns ending with this (e.g., 'US Equity')
    - max_daily_change: threshold for detecting split (e.g., 0.25 = 25%)

    Returns:
    - adjusted_nav_df: copy of nav_df with split-adjusted prices
    """
    df = nav_df.copy()
    etf_cols = [col for col in df.columns if col.endswith(etf_suffix)]

    for col in etf_cols:
        price = df[col].dropna()
        if len(price) < 2:
            continue

        # Compute daily returns
        returns = price.pct_change()

        # Identify likely split days: |return| > max_daily_change AND not due to market crash/rally
        # We assume true market moves rarely exceed ±25% in one day for broad ETFs
        split_flags = returns.abs() > max_daily_change

        if not split_flags.any():
            continue

        date_list = price.index[split_flags]
        adjusted_price = price.copy()

        # print(fetf_cols)
        # print(date_list)
        # print(returns[split_flags])

        # Work backward from end to start to apply adjustment factors
        for dt in date_list[::-1]:
            ratio = returns[dt]
            if ratio < 0:
                scale = round(1/(1+ratio))
                adjusted_price.loc[dt:] = adjusted_price.loc[dt:] * scale
            elif ratio > 0:
                scale = round(1+ratio)
                adjusted_price.loc[dt:] = adjusted_price.loc[dt:] / scale

        # Put back into df
        df.loc[adjusted_price.index, col] = adjusted_price

    return df


def get_proxy_data(client_name, risk_level):
    print(f'Import Data for {client_name}')
    proxy_info = pd.read_excel(f'{client_name}/proxy_info.xlsx', header=0)
    ticker_to_type = dict(zip(proxy_info['BBG Ticker'], proxy_info['代表指数名称']))
    nav_df = pd.read_csv(f'{client_name}/proxy_data.csv', index_col=0, parse_dates=True).ffill()
    nav_df = adjust_splits_in_nav(nav_df).rename(columns=ticker_to_type)

    # fx conversion
    fx_df = pd.read_csv(f'{client_name}/fx_data.csv', header=0, index_col=0, parse_dates=True).ffill()
    for idx in proxy_info.index:
        proxy_currency = proxy_info.loc[idx, 'Currency']
        proxy_ticker = proxy_info.loc[idx, '代表指数名称']
        if proxy_currency != 'USD':
            fx_col = f'{proxy_currency}USD Curncy'
            date_list = sorted(list(set(fx_df[fx_col].dropna().index.values) & set(nav_df[proxy_ticker].dropna().index.values)))
            nav_df[proxy_ticker] = nav_df.loc[date_list, proxy_ticker] / fx_df.loc[date_list, fx_col]

    # resample for weekly data
    nav_df = nav_df.resample('W-FRI').last()
    market_caps = pd.Series(index=proxy_info['代表指数名称'].values, data=proxy_info['Market Cap New'].values)  # Index: assets (indices)
    # Compute returns from NAV
    historical_returns = compute_returns(nav_df)

    # pre-process for selected risk level
    selected_class = proxy_info[proxy_info[risk_level] == 'Y']['代表指数名称']
    print(f"Client's Risk Level: {risk_level}")
    drop_classes_list = ['美股', '贵金属']
    for drop_class in drop_classes_list:
        selected_class = selected_class[selected_class != reverse_sub_class_map[drop_class]]
    print(f"Selected Asset Classes: \n {[sub_class_map.get(idx, idx) for idx in selected_class]}")

    # proxy replacement
    proxy_info_change = proxy_info.dropna(subset='New Proxy')
    for idx in proxy_info_change.index[:-1]:
        print(f"\n Change Underlying Proxy for {proxy_info_change.loc[idx, '资产子类']}")
        print(f"Old Proxy: {proxy_info_change.loc[idx, '代表指数名称']}")
        print(f"New Proxy: {proxy_info_change.loc[idx, 'New Proxy']}")

    historical_returns = historical_returns[selected_class]
    market_caps = market_caps[selected_class]

    return market_caps, historical_returns


def plot_matrix(cov_df: pd.DataFrame):
    """
    Plot CORRELATION matrix derived from a COVARIANCE matrix.
    Input: cov_df — covariance matrix (e.g., Sigma_bl from Black-Litterman)
    """
    # Convert covariance → correlation
    std = np.sqrt(np.diag(cov_df.values))
    corr = cov_df.values / np.outer(std, std)
    corr_df = pd.DataFrame(corr, index=cov_df.index, columns=cov_df.columns)

    # Map index names to sub-class labels
    corr_df = corr_df.rename(columns=sub_class_map, index=sub_class_map)

    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale='RdBu_r',
        zmid=0,
        text=np.around(corr_df.values, 3),
        texttemplate="%{text}",
        hoverongaps=False
    ))
    fig.update_layout(
        title="Implied Correlation Matrix (from Black-Litterman Covariance)",
        width=700,
        height=600
    )
    fig.show()


def plot_series(weights, series_type='Weights'):
    """
    Plot weights or expected returns using Plotly (interactive, polished).
    """
    # Filter small weights for clarity (optional)
    if series_type != 'Return':
        threshold = 0.005  # 0.5%
        weights_filtered = weights[weights >= threshold]
    else:
        weights_filtered = weights

    # Sort for better visualization
    weights_filtered = weights_filtered.sort_values(ascending=True)

    # Use asset sub-class names (already mapped in weights index)
    labels = weights_filtered.index
    values = weights_filtered.values

    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker_color='steelblue',
        text=[f"{v:.1%}" for v in values],
        textposition='auto',
        hovertemplate='%{y}: %{x:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Portfolio {series_type}',
        xaxis_title=series_type,
        xaxis_tickformat='.1%',
        height=max(400, len(labels) * 25),
        margin=dict(l=150, r=50, t=50, b=50)
    )

    fig.show()


def print_weight_summary(weights: pd.Series, title: str = "Portfolio"):
    """
    Print portfolio weights grouped by broad asset class and sub-class,
    with proper alignment for mixed Chinese/English text.
    """
    print(f"\n=== {title} Weight Summary ===")

    unknown = [sub for sub in weights.index if sub not in sub_class_to_broad_class]
    if unknown:
        print(f"⚠️ Warning: Unknown sub-classes: {unknown}")

    # Aggregate by broad class
    broad_class_weights = {}
    for sub_class, w in weights.items():
        if sub_class in sub_class_to_broad_class:
            broad = sub_class_to_broad_class[sub_class]
            broad_class_weights[broad] = broad_class_weights.get(broad, 0.0) + w

    # Enforce order
    broad_order = ['权益', '债券', '货币', '大宗商品', '另类投资']
    broad_class_weights = {k: broad_class_weights.get(k, 0.0) for k in broad_order}

    # --- Print Broad Asset Class (with alignment) ---
    print("\nBy Broad Asset Class:")
    total_check = 0.0
    for broad, w in broad_class_weights.items():
        total_check += w
        # Format: "权益    :  45.20%"
        label = f"{broad}"
        weight_str = f"{w:>7.2%}"
        # Align the colon at a fixed display width (e.g., 8)
        current_width = wcswidth(label)
        padding = ' ' * max(0, 8 - current_width)
        print(f"  {label}{padding}: {weight_str}")

    # Total line
    total_label = "Total"
    total_weight_str = f"{total_check:>7.2%}"
    total_current_width = wcswidth(total_label)
    total_padding = ' ' * max(0, 8 - total_current_width)
    print(f"  {total_label}{total_padding}: {total_weight_str}")

    # --- Print Sub-Class (with alignment) ---
    print("\nBy Sub-Class (non-zero):")
    sub_weights_sorted = weights[weights > 1e-6].sort_values(ascending=False)

    for sub, w in sub_weights_sorted.items():
        broad = sub_class_to_broad_class.get(sub, "Unknown")
        # Format the sub-class name with display-width-aware padding
        # Target total width for "sub (broad)" part: e.g., 35 display columns
        label = f"{sub} ({broad})"
        weight_str = f"{w:>7.2%}"

        # Calculate how much padding is needed to align weight_str
        current_width = wcswidth(label)
        target_width = 35  # adjust as needed
        if current_width < target_width:
            padding = ' ' * (target_width - current_width)
            line = label + padding + weight_str
        else:
            line = label + ' ' + weight_str
        print(line)


def plot_backtest(nav: pd.Series, title: str = "Portfolio Backtest"):
    """
    Plot NAV over time using Plotly.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nav.index,
        y=nav.values,
        mode='lines',
        name='Portfolio NAV',
        line=dict(color='darkblue', width=2)
    ))
    fig.update_layout(
        title=title,
        yaxis_title='Cumulative NAV (Base = 1.0)',
        xaxis_title='Date',
        hovermode='x unified'
    )
    fig.show()


def print_portfolio_history(portfolio_history: dict, top_n: int = 10):
    """
    Print portfolio weights at each rebalance date with aligned columns.
    """
    if not portfolio_history:
        print("No portfolio history to display.")
        return

    # Use wcwidth for alignment
    try:
        from wcwidth import wcswidth
    except ImportError:
        def wcswidth(s):
            return sum(2 if '\u4e00' <= c <= '\u9fff' else 1 for c in s)

    dates = sorted(portfolio_history.keys())
    print(f"\n=== Portfolio History ({len(dates)} Rebalances, Top {top_n} Asset Classes) ===")

    for date in dates:
        weights = portfolio_history[date]
        top_weights = weights[weights > 1e-6].sort_values(ascending=False).head(top_n)

        if top_weights.empty:
            continue

        print(f"\n📅 {date.date()}:")

        # Find max display width for asset names in this date
        max_asset_width = max((wcswidth(str(asset)) for asset in top_weights.index), default=20)

        for asset, w in top_weights.items():
            current_width = wcswidth(str(asset))
            padding = ' ' * (max_asset_width - current_width)
            print(f"  {asset}{padding}: {w:>8.2%}")

        # Total line
        total = weights.sum()
        total_label = "Total"
        total_width = wcswidth(total_label)
        total_padding = ' ' * (max_asset_width - total_width)
        print(f"  {total_label}{total_padding}: {total:>8.2%}")


def plot_portfolio_history(portfolio_history: dict, title: str = "Portfolio Weight Evolution"):
    """
    Plot portfolio weights over time as a stacked area chart (Plotly).

    Parameters:
    - portfolio_history: dict {date: pd.Series(weights)}
    - title: plot title
    """
    if not portfolio_history:
        print("No portfolio history to plot.")
        return

    # Convert to DataFrame: rows = dates, columns = assets
    df_list = []
    for date, weights in portfolio_history.items():
        # Fill missing assets with 0
        df_list.append(weights.rename(date))
    history_df = pd.concat(df_list, axis=1).T
    history_df = history_df.fillna(0)

    # Optional: filter out tiny or zero-variance assets
    min_weight_threshold = 0.0001  # 1%
    avg_weights = history_df.mean()
    significant_assets = avg_weights[avg_weights >= min_weight_threshold].index
    if len(significant_assets) == 0:
        significant_assets = history_df.columns  # show all if none above threshold
    plot_df = history_df[significant_assets]

    # Transpose for Plotly: columns = assets, index = dates
    plot_df = plot_df.T

    # Create stacked area chart
    fig = go.Figure()
    for asset in plot_df.index:
        fig.add_trace(go.Scatter(
            x=plot_df.columns,
            y=plot_df.loc[asset],
            mode='lines',
            stackgroup='one',
            name=asset,
            groupnorm=''  # no normalization
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Rebalance Date",
        yaxis_title="Portfolio Weight",
        yaxis_tickformat='.1%',
        hovermode="x unified",
        legend_title="Asset Sub-Class",
        height=600
    )
    fig.show()


def print_view_sensitivity_results(baseline_weights, sensitivity_results, stats_baseline):
    print("\n" + "=" * 60)
    print("VIEW SENSITIVITY ANALYSIS (Weight + Performance Impact)")
    print("=" * 60)

    print("\n.BASELINE PORTFOLIO PERFORMANCE:")
    for k, v in stats_baseline.items():
        if k == 'Max Drawdown':
            print(f"  {k:<25}: {v:>8.2%}")
        elif 'Ratio' in k:
            print(f"  {k:<25}: {v:>8.2f}")
        else:
            print(f"  {k:<25}: {v:>8.2%}")

    if not sensitivity_results:
        return

    print(f"\n.TOP {len(sensitivity_results)} VIEWS BY WEIGHT IMPACT:")
    print("-" * 60)

    # Use wcwidth for alignment
    try:
        from wcwidth import wcswidth
    except ImportError:
        def wcswidth(s):
            return sum(2 if '\u4e00' <= c <= '\u9fff' else 1 for c in s)

    for i, res in enumerate(sensitivity_results, 1):
        print(f"{i:2}. {res['view_desc']}")
        print(f"     Weight Impact: {res['total_weight_impact']:>7.2%}")

        # Performance impact
        perf = res['perf_impact']
        print(f"     Return Δ: {perf['Return']:+7.2%} | Vol Δ: {perf['Volatility']:+7.2%} | Sharpe Δ: {perf['Sharpe']:+7.2f}")

        print(f"     Top 6 weight changes (view added):")
        # Get top 6 by absolute impact
        impact = res['view_impact']
        top6_abs = impact.abs().sort_values(ascending=False).head(6)
        top6 = impact.reindex(top6_abs.index)

        # Find max display width for alignment
        max_width = max((wcswidth(str(asset)) for asset in top6.index), default=20)

        for asset, delta in top6.items():
            # Pad asset name to max display width
            current_width = wcswidth(str(asset))
            padding = ' ' * (max_width - current_width)
            print(f"       {asset}{padding}: {delta:+8.2%}")
        print()


def plot_view_sensitivity(sensitivity_results, top_k: int = 3):
    """
    Plot top weight changes for the most impactful views.
    """
    if not sensitivity_results:
        return

    top_views = sensitivity_results[:top_k]
    fig = go.Figure()

    for res in top_views:
        impact = res['view_impact']
        # impact = impact[impact.abs() > 0]
        top_changes = impact.abs().sort_values(ascending=False).head(3)
        top_changes = impact.reindex(top_changes.index)  # keep signed values

        fig.add_trace(go.Bar(
            x=top_changes.index,
            y=top_changes.values,
            name=res['view_desc'][:40] + "...",
            text=[f"{v:+.2%}" for v in top_changes.values],
            textposition='auto'
        ))

    fig.update_layout(
        title=f"Top Weight Changes Due to Views (Top {top_k})",
        barmode='group',
        yaxis_tickformat='.2%',
        height=500,
        xaxis_tickangle=-45,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=100),  # top margin in pixels
        # Optional: move title slightly down if needed
        title_x=0.5,  # center title
        title_y=0.98  # title position (1.0 = very top)
    )
    fig.show()


def compare_portfolios(
        weights_a: pd.Series,
        weights_b: pd.Series,
        returns_df: pd.DataFrame,
        window: float,
        name_a: str = "Portfolio A",
        name_b: str = "Portfolio B"
):
    """
    Compare two portfolios:
    1. Show weight allocation
    2. Show backtested NAV
    3. Print performance metrics

    Returns NAVs, stats.
    """
    # 1. Backtest both portfolios (needed for NAV and stats)
    nav_a, stats_a = backtest_portfolio(weights_a, returns_df, window, rebalance=False)
    nav_b, stats_b = backtest_portfolio(weights_b, returns_df, window, rebalance=False)

    # Align NAVs
    all_dates = nav_a.index.union(nav_b.index)
    nav_a = nav_a.reindex(all_dates, method='pad')
    nav_b = nav_b.reindex(all_dates, method='pad')

    # 2. === FIRST: Plot Weight Comparison ===
    all_assets = weights_a.index.union(weights_b.index)
    w_a = weights_a.reindex(all_assets, fill_value=0.0)
    w_b = weights_b.reindex(all_assets, fill_value=0.0)

    # Filter small weights
    min_weight = 0.005
    significant = (w_a.abs() >= min_weight) | (w_b.abs() >= min_weight)
    w_a = w_a[significant]
    w_b = w_b[significant]

    # Sort by average weight
    avg_weights = (w_a + w_b) / 2
    sort_order = avg_weights.sort_values(ascending=False).index
    w_a = w_a.reindex(sort_order)
    w_b = w_b.reindex(sort_order)

    fig_weights = go.Figure()
    fig_weights.add_trace(go.Bar(
        x=w_a.index,
        y=w_a.values,
        name=name_a,
        marker_color='steelblue'
    ))
    fig_weights.add_trace(go.Bar(
        x=w_b.index,
        y=w_b.values,
        name=name_b,
        marker_color='indianred'
    ))
    fig_weights.update_layout(
        title=f"{name_a} vs {name_b} — Portfolio Weights",
        yaxis_title="Weight",
        yaxis_tickformat='.1%',
        barmode='group',
        height=500,
        xaxis_tickangle=-45,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=100)
    )
    fig_weights.show()

    # 3. === SECOND: Plot NAV Comparison ===
    fig_nav = go.Figure()
    fig_nav.add_trace(go.Scatter(x=nav_a.index, y=nav_a.values, mode='lines', name=name_a))
    fig_nav.add_trace(go.Scatter(x=nav_b.index, y=nav_b.values, mode='lines', name=name_b))
    fig_nav.update_layout(
        title=f"{name_a} vs {name_b} — Cumulative NAV",
        yaxis_title="Cumulative NAV (Base = 1.0)",
        hovermode="x unified",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=100)
    )
    fig_nav.show()

    # 4. === THIRD: Print Performance Metrics ===
    print(f"\n=== {name_a} vs {name_b} Performance Metrics ===")
    print(f"{'Metric':<25} {name_a:>12} {name_b:>12} {'Δ (A-B)':>12}")
    print("-" * 70)
    for metric in ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown']:
        a_val = stats_a[metric]
        b_val = stats_b[metric]
        diff = a_val - b_val
        if metric == 'Max Drawdown':
            print(f"{metric:<25} {a_val:>11.2%} {b_val:>11.2%} {diff:>11.2%}")
        elif 'Ratio' in metric:
            print(f"{metric:<25} {a_val:>11.2f} {b_val:>11.2f} {diff:>11.2f}")
        else:
            print(f"{metric:<25} {a_val:>11.2%} {b_val:>11.2%} {diff:>11.2%}")

    return nav_a, nav_b, stats_a, stats_b


