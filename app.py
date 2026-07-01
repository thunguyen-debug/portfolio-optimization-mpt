"""
Portfolio Optimizer Dashboard
Modern fintech design inspired by Robinhood
Institutional-grade mean-variance optimization

Author: Thu Nguyen
GitHub: github.com/thunguyen-debug
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

css = """
<style>
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    body, [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
    }
    
    [data-testid="stSidebar"] {
        background-color: #fafbfc;
        border-right: 1px solid #e5e7eb;
    }
    
    h1 {
        color: #111827;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    h2 {
        color: #1f2937;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1.25rem;
        letter-spacing: -0.5px;
    }
    
    h3 {
        color: #374151;
        font-size: 1.125rem;
        font-weight: 600;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    
    [data-testid="stMetricDelta"] {
        color: #10b981;
    }
    
    [data-testid="stDataFrame"] {
        font-size: 0.95rem;
    }
    
    [data-testid="stDataFrameResizable"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        overflow: hidden;
    }
    
    [data-testid="stDivider"] {
        margin: 2rem 0;
        border-color: #e5e7eb;
    }
    
    .stButton > button {
        background-color: #10b981;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        font-size: 0.95rem;
    }
    
    .stButton > button:hover {
        background-color: #059669;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .stSelectbox, .stMultiSelect, .stSlider, .stRadio {
        margin-bottom: 1.5rem;
    }
    
    .metric-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    [data-testid="stAlert"] {
        border-radius: 8px;
        padding: 1rem;
    }
</style>
"""

st.markdown(css, unsafe_allow_html=True)

@st.cache_data
def generate_sample_data(tickers, days=1825):
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    params = {
        'SPY': {'mu': 0.10, 'sigma': 0.15},
        'EFA': {'mu': 0.08, 'sigma': 0.18},
        'EEM': {'mu': 0.09, 'sigma': 0.22},
        'BND': {'mu': 0.03, 'sigma': 0.05},
        'BNDX': {'mu': 0.03, 'sigma': 0.06},
        'VNQ': {'mu': 0.08, 'sigma': 0.20},
        'GSG': {'mu': 0.04, 'sigma': 0.15},
        'GLD': {'mu': 0.06, 'sigma': 0.12}
    }
    
    data = {}
    for ticker in tickers:
        if ticker in params:
            p = params[ticker]
        else:
            p = {'mu': 0.07, 'sigma': 0.15}
        
        returns = np.random.normal(p['mu']/252, p['sigma']/np.sqrt(252), days)
        price = 100 * np.exp(np.cumsum(returns))
        data[ticker] = price
    
    return pd.DataFrame(data, index=dates)

with st.sidebar:
    st.markdown("### Settings")
    st.divider()
    
    data_source = st.radio(
        "Data Source",
        ["Historical (Yahoo Finance)", "Sample Data (Demo)", "Upload CSV"],
        label_visibility="collapsed"
    )
    
    if data_source == "Historical (Yahoo Finance)" or data_source == "Sample Data (Demo)":
        available_assets = {
            'US Equities': 'SPY',
            'International Equities': 'EFA',
            'Emerging Markets': 'EEM',
            'US Bonds': 'BND',
            'International Bonds': 'BNDX',
            'Real Estate (REITs)': 'VNQ',
            'Commodities': 'GSG',
            'Gold': 'GLD'
        }
        
        selected_assets = st.multiselect(
            "Assets",
            list(available_assets.keys()),
            default=['US Equities', 'US Bonds', 'International Equities'],
            label_visibility="collapsed"
        )
        
        tickers = [available_assets[asset] for asset in selected_assets]
        
        time_period = st.selectbox(
            "Period",
            ["1 Year", "3 Years", "5 Years", "10 Years"],
            index=2,
            label_visibility="collapsed"
        )
        
        period_map = {
            "1 Year": 1 * 365,
            "3 Years": 3 * 365,
            "5 Years": 5 * 365,
            "10 Years": 10 * 365
        }
        
        days_back = period_map[time_period]
        
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'], label_visibility="collapsed")
        tickers = None
    
    st.divider()
    
    st.markdown("### Optimization")
    
    rf_rate = st.slider(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=4.5,
        step=0.1,
        label_visibility="collapsed"
    ) / 100
    
    cov_method = st.selectbox(
        "Covariance",
        ["Sample", "Ledoit-Wolf Shrinkage"],
        label_visibility="collapsed"
    )

@st.cache_data
def load_data(tickers, days_back):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    data_list = []
    successful_tickers = []
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df, pd.DataFrame):
                price_series = df['Adj Close'].copy()
            else:
                price_series = df.copy()
            price_series.name = ticker
            data_list.append(price_series)
            successful_tickers.append(ticker)
        except:
            pass
    
    if len(data_list) == 0:
        return None, None
    
    data = pd.concat(data_list, axis=1)
    data.columns = successful_tickers
    data = data.dropna()
    
    return data, successful_tickers

@st.cache_data
def compute_statistics(data):
    daily_returns = data.pct_change().dropna()
    annual_returns = daily_returns.mean() * 252
    annual_vol = daily_returns.std() * np.sqrt(252)
    cov_matrix = daily_returns.cov() * 252
    
    return daily_returns, annual_returns, annual_vol, cov_matrix

def estimate_covariance_robust(daily_returns):
    try:
        lw = LedoitWolf()
        cov_shrink, _ = lw.fit(daily_returns)
        return pd.DataFrame(cov_shrink * 252, index=daily_returns.columns, columns=daily_returns.columns)
    except:
        return daily_returns.cov() * 252

def portfolio_stats(weights, returns, cov_matrix, rf_rate):
    portfolio_return = np.sum(weights * returns)
    portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_vol = np.sqrt(portfolio_var)
    sharpe_ratio = (portfolio_return - rf_rate) / portfolio_vol if portfolio_vol > 0 else 0
    return portfolio_return, portfolio_vol, sharpe_ratio

def negative_sharpe(weights, returns, cov_matrix, rf_rate):
    return -portfolio_stats(weights, returns, cov_matrix, rf_rate)[2]

def portfolio_volatility(weights, returns, cov_matrix, rf_rate):
    return portfolio_stats(weights, returns, cov_matrix, rf_rate)[1]

def optimize_portfolio(returns, cov_matrix, rf_rate, objective='sharpe'):
    n = len(returns)
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    
    if objective == 'sharpe':
        result = minimize(negative_sharpe, x0=np.array([1/n]*n), args=(returns, cov_matrix, rf_rate),
                         method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000})
    elif objective == 'min_var':
        result = minimize(portfolio_volatility, x0=np.array([1/n]*n), args=(returns, cov_matrix, rf_rate),
                         method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x if result.success else None

def risk_parity_allocation(cov_matrix):
    n = len(cov_matrix)
    weights = np.array([1.0 / n] * n)
    
    for _ in range(100):
        port_var = np.dot(weights, np.dot(cov_matrix, weights))
        port_vol = np.sqrt(port_var)
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / (port_vol + 1e-10)
        weights = (1.0 / (risk_contrib + 1e-10)) / np.sum(1.0 / (risk_contrib + 1e-10))
    
    return weights

def compute_efficient_frontier(returns, cov_matrix, rf_rate, n_points=80):
    min_return = returns.min() * 0.8
    max_return = returns.max() * 1.2
    target_returns = np.linspace(min_return, max_return, n_points)
    
    frontier_vols = []
    frontier_returns = []
    
    for target_ret in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, returns) - target_ret}
        ]
        
        n = len(returns)
        bounds = tuple((0, 1) for _ in range(n))
        
        result = minimize(portfolio_volatility, x0=np.array([1/n]*n), args=(returns, cov_matrix, rf_rate),
                         method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000})
        
        if result.success:
            _, vol, _ = portfolio_stats(result.x, returns, cov_matrix, rf_rate)
            frontier_vols.append(vol)
            frontier_returns.append(target_ret)
    
    return np.array(frontier_returns), np.array(frontier_vols)

st.markdown("# Portfolio Optimizer")
st.markdown("Institutional-grade portfolio optimization")

if data_source == "Historical (Yahoo Finance)":
    if not tickers:
        st.warning("Select at least one asset from sidebar")
        st.stop()
    
    with st.spinner("Loading data..."):
        try:
            data, successful_tickers = load_data(tickers, days_back)
            if data is None or len(data) == 0:
                st.info("Using sample data")
                data = generate_sample_data(tickers, days_back)
                successful_tickers = tickers
            tickers = successful_tickers
        except:
            st.info("Using sample data")
            data = generate_sample_data(tickers, days_back)
            
elif data_source == "Sample Data (Demo)":
    if not tickers:
        st.warning("Select at least one asset from sidebar")
        st.stop()
    st.info("Sample data")
    data = generate_sample_data(tickers, days_back)
    
else:
    if uploaded_file is None:
        st.warning("Upload a CSV file")
        st.stop()
    
    try:
        data = pd.read_csv(uploaded_file, index_col=0)
        tickers = list(data.columns)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.stop()

try:
    daily_returns, annual_returns, annual_vol, cov_matrix = compute_statistics(data)
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.stop()

if cov_method == "Ledoit-Wolf Shrinkage":
    try:
        cov_matrix = estimate_covariance_robust(daily_returns)
    except:
        pass

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Assets", len(tickers))
with col2:
    st.metric("Period", time_period)
with col3:
    st.metric("Risk-Free Rate", f"{rf_rate*100:.1f}%")

st.divider()

st.markdown("## Asset Summary")

asset_summary = pd.DataFrame({
    'Asset': tickers,
    'Return %': (annual_returns.values * 100).round(2),
    'Volatility %': (annual_vol.values * 100).round(2),
    'Sharpe': ((annual_returns.values - rf_rate) / annual_vol.values).round(3)
})

st.dataframe(asset_summary, use_container_width=True, hide_index=True)

st.divider()

try:
    min_var_weights = optimize_portfolio(annual_returns.values, cov_matrix.values, rf_rate, 'min_var')
    max_sharpe_weights = optimize_portfolio(annual_returns.values, cov_matrix.values, rf_rate, 'sharpe')
    rp_weights = risk_parity_allocation(cov_matrix.values)
    ew_weights = np.array([1.0/len(tickers)]*len(tickers))
    
    if min_var_weights is None or max_sharpe_weights is None:
        st.error("Could not compute optimal portfolios")
        st.stop()
    
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.stop()

def get_portfolio_stats(weights, name):
    ret, vol, sharpe = portfolio_stats(weights, annual_returns.values, cov_matrix.values, rf_rate)
    return {
        'Strategy': name,
        'Return %': ret * 100,
        'Volatility %': vol * 100,
        'Sharpe': sharpe
    }

results = [
    get_portfolio_stats(min_var_weights, 'Min Variance'),
    get_portfolio_stats(max_sharpe_weights, 'Max Sharpe'),
    get_portfolio_stats(rp_weights, 'Risk Parity'),
    get_portfolio_stats(ew_weights, 'Equal Weight')
]

results_df = pd.DataFrame(results)

st.markdown("## Optimization Results")
st.dataframe(results_df.set_index('Strategy'), use_container_width=True)

st.divider()

st.markdown("## Allocations")

col1, col2, col3, col4 = st.columns(4)

portfolios = [
    (min_var_weights, 'Min Variance', col1),
    (max_sharpe_weights, 'Max Sharpe', col2),
    (rp_weights, 'Risk Parity', col3),
    (ew_weights, 'Equal Weight', col4)
]

for weights, name, col in portfolios:
    with col:
        allocation = pd.DataFrame({'Asset': tickers, 'Weight': (weights * 100).round(2)})
        allocation = allocation[allocation['Weight'] > 0.1].sort_values('Weight', ascending=False)
        
        fig = go.Figure(data=[go.Pie(labels=allocation['Asset'], values=allocation['Weight'], 
                                     textinfo='label+percent', marker=dict(
                                         colors=['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6']
                                     ))])
        fig.update_layout(title=name, height=360, margin=dict(l=0, r=0, t=40, b=0), 
                         font=dict(size=11), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

st.divider()

st.markdown("## Efficient Frontier")

try:
    frontier_returns, frontier_vols = compute_efficient_frontier(annual_returns.values, cov_matrix.values, rf_rate, n_points=100)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=frontier_vols*100, y=frontier_returns*100, mode='lines', 
                            name='Frontier', line=dict(color='#10b981', width=3),
                            hovertemplate='<b>Volatility:</b> %{x:.1f}%<br><b>Return:</b> %{y:.1f}%<extra></extra>'))
    
    for i, ticker in enumerate(tickers):
        fig.add_trace(go.Scatter(x=[annual_vol.values[i]*100], y=[annual_returns.values[i]*100], 
                                mode='markers', name=ticker, marker=dict(size=12, opacity=0.7),
                                hovertemplate=f'<b>{ticker}</b><br>Vol: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<extra></extra>',
                                showlegend=True))
    
    optimized = [
        (min_var_weights, 'Min Var', '#3b82f6'),
        (max_sharpe_weights, 'Max Sharpe', '#ef4444'),
        (rp_weights, 'Risk Par', '#f59e0b'),
        (ew_weights, 'Eq Weight', '#8b5cf6')
    ]
    
    for weights, label, color in optimized:
        ret, vol, sharpe = portfolio_stats(weights, annual_returns.values, cov_matrix.values, rf_rate)
        fig.add_trace(go.Scatter(x=[vol*100], y=[ret*100], mode='markers', name=label,
                                marker=dict(size=20, symbol='star', color=color, 
                                           line=dict(color='white', width=2)),
                                hovertemplate=f'<b>{label}</b><br>Vol: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<br>Sharpe: {sharpe:.3f}<extra></extra>',
                                showlegend=True))
    
    if len(frontier_vols) > 0:
        max_vol = frontier_vols.max()
        cal_vols = np.linspace(0, max_vol*1.2, 100)
        sharpe_max = portfolio_stats(max_sharpe_weights, annual_returns.values, cov_matrix.values, rf_rate)[2]
        cal_returns = rf_rate + sharpe_max * cal_vols
        
        fig.add_trace(go.Scatter(x=cal_vols*100, y=cal_returns*100, mode='lines', 
                                name='CAL', line=dict(color='#9ca3af', width=2, dash='dash'),
                                hovertemplate='Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'))
    
    fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Annual Volatility (%)',
        yaxis_title='Annual Return (%)',
        height=650,
        hovermode='closest',
        template='plotly_white',
        font=dict(size=11),
        plot_bgcolor='#f9fafb',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"Error: {str(e)}")

st.divider()

st.markdown("## Risk Attribution")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Max Sharpe")
    try:
        port_vol_ms = portfolio_stats(max_sharpe_weights, annual_returns.values, cov_matrix.values, rf_rate)[1]
        marginal_contrib_ms = np.dot(cov_matrix.values, max_sharpe_weights)
        rc_ms = max_sharpe_weights * marginal_contrib_ms / (port_vol_ms + 1e-10)
        
        rc_df_ms = pd.DataFrame({'Asset': tickers, 'Contribution %': (rc_ms * 100).round(2)})
        rc_df_ms = rc_df_ms.sort_values('Contribution %', ascending=True)
        
        fig_rc = px.bar(rc_df_ms, y='Asset', x='Contribution %', orientation='h',
                       color='Contribution %', color_continuous_scale='Blues',
                       showlegend=False)
        fig_rc.update_layout(height=320, template='plotly_white', font=dict(size=11),
                            plot_bgcolor='#f9fafb')
        st.plotly_chart(fig_rc, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")

with col2:
    st.markdown("### Risk Parity")
    try:
        port_vol_rp = portfolio_stats(rp_weights, annual_returns.values, cov_matrix.values, rf_rate)[1]
        marginal_contrib_rp = np.dot(cov_matrix.values, rp_weights)
        rc_rp = rp_weights * marginal_contrib_rp / (port_vol_rp + 1e-10)
        
        rc_df_rp = pd.DataFrame({'Asset': tickers, 'Contribution %': (rc_rp * 100).round(2)})
        rc_df_rp = rc_df_rp.sort_values('Contribution %', ascending=True)
        
        fig_rc = px.bar(rc_df_rp, y='Asset', x='Contribution %', orientation='h',
                       color='Contribution %', color_continuous_scale='Greens',
                       showlegend=False)
        fig_rc.update_layout(height=320, template='plotly_white', font=dict(size=11),
                            plot_bgcolor='#f9fafb')
        st.plotly_chart(fig_rc, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.divider()

st.markdown("## Correlations")

try:
    corr_matrix = daily_returns.corr()
    
    fig_corr = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=tickers, y=tickers, 
                                         colorscale='RdBu', zmid=0, text=np.round(corr_matrix.values, 2),
                                         texttemplate='%{text:.2f}', textfont={"size": 11},
                                         colorbar=dict(title="Correlation", thickness=15)))
    
    fig_corr.update_layout(title='Correlation Matrix', height=500, template='plotly_white',
                          font=dict(size=11))
    
    st.plotly_chart(fig_corr, use_container_width=True)
except Exception as e:
    st.error(f"Error: {str(e)}")

st.divider()

st.markdown("## Downloads")

try:
    download_data = pd.DataFrame({
        'Strategy': ['Min Variance', 'Max Sharpe', 'Risk Parity', 'Equal Weight'],
        'Return %': results_df['Return %'].round(2),
        'Volatility %': results_df['Volatility %'].round(2),
        'Sharpe': results_df['Sharpe'].round(3)
    })
    
    allocations_download = pd.concat([
        pd.DataFrame({'Strategy': 'Min Variance', 'Asset': tickers, 'Weight %': (min_var_weights*100).round(2)}),
        pd.DataFrame({'Strategy': 'Max Sharpe', 'Asset': tickers, 'Weight %': (max_sharpe_weights*100).round(2)}),
        pd.DataFrame({'Strategy': 'Risk Parity', 'Asset': tickers, 'Weight %': (rp_weights*100).round(2)}),
        pd.DataFrame({'Strategy': 'Equal Weight', 'Asset': tickers, 'Weight %': (ew_weights*100).round(2)})
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv1 = download_data.to_csv(index=False)
        st.download_button(label="Performance Metrics", data=csv1, file_name="metrics.csv", mime="text/csv")
    
    with col2:
        csv2 = allocations_download.to_csv(index=False)
        st.download_button(label="Allocations", data=csv2, file_name="allocations.csv", mime="text/csv")
except Exception as e:
    st.error(f"Error: {str(e)}")

st.divider()

st.markdown("""
---
Built by Thu Nguyen | Github: github.com/thunguyen-debug | Email: thunguyen5260@gmail.com
""")
