"""
Portfolio Optimizer Dashboard
Institutional-grade mean-variance optimization with interactive interface
Built for recruiting, suitable for production use

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
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def generate_sample_data(tickers, days=1825):
    """Generate realistic sample data for demonstration"""
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

st.sidebar.markdown("# ⚙️ Portfolio Settings")
st.sidebar.markdown("---")

data_source = st.sidebar.radio(
    "📊 Data Source",
    ["Historical (Yahoo Finance)", "Sample Data (Demo)", "Upload CSV"],
    help="Use historical data from Yahoo Finance, sample data, or upload your own CSV"
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
    
    selected_assets = st.sidebar.multiselect(
        "🏢 Select Assets",
        list(available_assets.keys()),
        default=['US Equities', 'US Bonds', 'International Equities'],
        help="Choose which asset classes to include"
    )
    
    tickers = [available_assets[asset] for asset in selected_assets]
    
    time_period = st.sidebar.selectbox(
        "📅 Historical Period",
        ["1 Year", "3 Years", "5 Years", "10 Years"],
        index=2,
        help="How much historical data to use for analysis"
    )
    
    period_map = {
        "1 Year": 1 * 365,
        "3 Years": 3 * 365,
        "5 Years": 5 * 365,
        "10 Years": 10 * 365
    }
    
    days_back = period_map[time_period]
    
else:
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload CSV (returns data)",
        type=['csv'],
        help="CSV with date index and asset columns"
    )
    tickers = None

st.sidebar.markdown("### 📈 Optimization Parameters")
rf_rate = st.sidebar.slider(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=4.5,
    step=0.1,
    help="Current 10-year US Treasury yield"
) / 100

cov_method = st.sidebar.selectbox(
    "Covariance Estimation",
    ["Sample", "Ledoit-Wolf Shrinkage"],
    help="Ledoit-Wolf is more robust, especially for small samples"
)

@st.cache_data
def load_data(tickers, days_back):
    """Download historical price data from Yahoo Finance"""
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
        except Exception as e:
            pass
    
    if len(data_list) == 0:
        return None, None
    
    data = pd.concat(data_list, axis=1)
    data.columns = successful_tickers
    data = data.dropna()
    
    return data, successful_tickers

@st.cache_data
def compute_statistics(data):
    """Compute returns, volatility, and covariance"""
    daily_returns = data.pct_change().dropna()
    annual_returns = daily_returns.mean() * 252
    annual_vol = daily_returns.std() * np.sqrt(252)
    cov_matrix = daily_returns.cov() * 252
    
    return daily_returns, annual_returns, annual_vol, cov_matrix

def estimate_covariance_robust(daily_returns):
    """Use Ledoit-Wolf shrinkage for robust covariance estimation"""
    try:
        lw = LedoitWolf()
        cov_shrink, _ = lw.fit(daily_returns)
        return pd.DataFrame(
            cov_shrink * 252,
            index=daily_returns.columns,
            columns=daily_returns.columns
        )
    except:
        return daily_returns.cov() * 252

def portfolio_stats(weights, returns, cov_matrix, rf_rate):
    """Calculate portfolio statistics"""
    portfolio_return = np.sum(weights * returns)
    portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_vol = np.sqrt(portfolio_var)
    sharpe_ratio = (portfolio_return - rf_rate) / portfolio_vol if portfolio_vol > 0 else 0
    return portfolio_return, portfolio_vol, sharpe_ratio

def negative_sharpe(weights, returns, cov_matrix, rf_rate):
    """Objective function to maximize Sharpe ratio"""
    return -portfolio_stats(weights, returns, cov_matrix, rf_rate)[2]

def portfolio_volatility(weights, returns, cov_matrix, rf_rate):
    """Objective function to minimize volatility"""
    return portfolio_stats(weights, returns, cov_matrix, rf_rate)[1]

def optimize_portfolio(returns, cov_matrix, rf_rate, objective='sharpe'):
    """Run portfolio optimization"""
    n = len(returns)
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    
    if objective == 'sharpe':
        result = minimize(
            negative_sharpe,
            x0=np.array([1/n]*n),
            args=(returns, cov_matrix, rf_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
    elif objective == 'min_var':
        result = minimize(
            portfolio_volatility,
            x0=np.array([1/n]*n),
            args=(returns, cov_matrix, rf_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
    
    return result.x if result.success else None

def risk_parity_allocation(cov_matrix):
    """Compute risk-parity weights (equal risk contribution)"""
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
    """Generate efficient frontier"""
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
        
        result = minimize(
            portfolio_volatility,
            x0=np.array([1/n]*n),
            args=(returns, cov_matrix, rf_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            _, vol, _ = portfolio_stats(result.x, returns, cov_matrix, rf_rate)
            frontier_vols.append(vol)
            frontier_returns.append(target_ret)
    
    return np.array(frontier_returns), np.array(frontier_vols)

st.markdown("# 🎯 Portfolio Optimizer Dashboard")
st.markdown("Institutional-grade mean-variance optimization & efficient frontier analysis")

if data_source == "Historical (Yahoo Finance)":
    if not tickers:
        st.warning("⚠️ Please select at least one asset from the sidebar")
        st.stop()
    
    with st.spinner("📥 Loading historical data..."):
        try:
            data, successful_tickers = load_data(tickers, days_back)
            if data is None or len(data) == 0:
                st.warning("⚠️ Could not download from Yahoo Finance. Using sample data instead...")
                data = generate_sample_data(tickers, days_back)
                successful_tickers = tickers
            tickers = successful_tickers
        except Exception as e:
            st.warning("⚠️ Using sample data for demonstration...")
            data = generate_sample_data(tickers, days_back)
            
elif data_source == "Sample Data (Demo)":
    if not tickers:
        st.warning("⚠️ Please select at least one asset from the sidebar")
        st.stop()
    
    st.info("📊 Using realistic sample data for demonstration")
    data = generate_sample_data(tickers, days_back)
    
else:
    if uploaded_file is None:
        st.warning("⚠️ Please upload a CSV file")
        st.stop()
    
    try:
        data = pd.read_csv(uploaded_file, index_col=0)
        tickers = list(data.columns)
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {str(e)}")
        st.stop()

try:
    daily_returns, annual_returns, annual_vol, cov_matrix = compute_statistics(data)
except Exception as e:
    st.error(f"❌ Error computing statistics: {str(e)}")
    st.stop()

if cov_method == "Ledoit-Wolf Shrinkage":
    try:
        cov_matrix = estimate_covariance_robust(daily_returns)
    except Exception as e:
        st.warning(f"⚠️ Using sample covariance instead")

st.markdown("---")
st.markdown("## 📊 Asset Class Summary")

asset_summary = pd.DataFrame({
    'Asset': tickers,
    'Annual Return': (annual_returns.values * 100).round(2),
    'Annual Volatility': (annual_vol.values * 100).round(2),
    'Sharpe Ratio': ((annual_returns.values - rf_rate) / annual_vol.values).round(3)
})

col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(asset_summary, use_container_width=True, hide_index=True)
with col2:
    st.metric("Total Assets", len(tickers))
    st.metric("Analysis Period", f"{time_period}")
    st.metric("Risk-Free Rate", f"{rf_rate*100:.2f}%")

st.markdown("---")
st.markdown("## 🚀 Portfolio Optimization")

st.info("Computing optimal portfolios...")

try:
    min_var_weights = optimize_portfolio(annual_returns.values, cov_matrix.values, rf_rate, 'min_var')
    max_sharpe_weights = optimize_portfolio(annual_returns.values, cov_matrix.values, rf_rate, 'sharpe')
    rp_weights = risk_parity_allocation(cov_matrix.values)
    ew_weights = np.array([1.0/len(tickers)]*len(tickers))
    
    if min_var_weights is None or max_sharpe_weights is None:
        st.error("❌ Could not compute optimal portfolios.")
        st.stop()
    
except Exception as e:
    st.error(f"❌ Error in portfolio optimization: {str(e)}")
    st.stop()

def get_portfolio_stats(weights, name):
    ret, vol, sharpe = portfolio_stats(weights, annual_returns.values, cov_matrix.values, rf_rate)
    return {
        'Strategy': name,
        'Return (%)': ret * 100,
        'Volatility (%)': vol * 100,
        'Sharpe Ratio': sharpe,
        'Return/Risk': (ret / vol) if vol > 0 else 0
    }

results = [
    get_portfolio_stats(min_var_weights, 'Minimum Variance'),
    get_portfolio_stats(max_sharpe_weights, 'Maximum Sharpe'),
    get_portfolio_stats(rp_weights, 'Risk Parity'),
    get_portfolio_stats(ew_weights, 'Equal Weight')
]

results_df = pd.DataFrame(results)

st.markdown("### Portfolio Optimization Results")
st.dataframe(results_df.set_index('Strategy'), use_container_width=True)

st.markdown("---")
st.markdown("## 📈 Portfolio Allocations")

col1, col2, col3, col4 = st.columns(4)

portfolios = [
    (min_var_weights, 'Minimum Variance', col1),
    (max_sharpe_weights, 'Maximum Sharpe', col2),
    (rp_weights, 'Risk Parity', col3),
    (ew_weights, 'Equal Weight', col4)
]

for weights, name, col in portfolios:
    with col:
        allocation = pd.DataFrame({
            'Asset': tickers,
            'Weight': (weights * 100).round(2)
        })
        allocation = allocation[allocation['Weight'] > 0.1]
        
        fig = go.Figure(data=[go.Pie(
            labels=allocation['Asset'],
            values=allocation['Weight'],
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>'
        )])
        fig.update_layout(
            title=name,
            height=400,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("## 📊 Efficient Frontier Analysis")

st.info("Computing efficient frontier...")

try:
    frontier_returns, frontier_vols = compute_efficient_frontier(
        annual_returns.values, cov_matrix.values, rf_rate, n_points=80
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=frontier_vols * 100,
        y=frontier_returns * 100,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='#1f77b4', width=3),
        hovertemplate='Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
    ))
    
    for i, ticker in enumerate(tickers):
        fig.add_trace(go.Scatter(
            x=[annual_vol.values[i] * 100],
            y=[annual_returns.values[i] * 100],
            mode='markers+text',
            name=ticker,
            marker=dict(size=10, opacity=0.7),
            text=[ticker],
            textposition='top center',
            hovertemplate=f'<b>{ticker}</b><br>Vol: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<extra></extra>'
        ))
    
    optimized = [
        (min_var_weights, 'Min Variance', '#ff7f0e'),
        (max_sharpe_weights, 'Max Sharpe', '#2ca02c'),
        (rp_weights, 'Risk Parity', '#d62728'),
        (ew_weights, 'Equal Weight', '#9467bd')
    ]
    
    for weights, name, color in optimized:
        ret, vol, sharpe = portfolio_stats(weights, annual_returns.values, cov_matrix.values, rf_rate)
        fig.add_trace(go.Scatter(
            x=[vol * 100],
            y=[ret * 100],
            mode='markers+text',
            name=name,
            marker=dict(size=15, symbol='star', color=color),
            text=[name],
            textposition='top center',
            hovertemplate=f'<b>{name}</b><br>Vol: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<br>Sharpe: {sharpe:.3f}<extra></extra>'
        ))
    
    if len(frontier_vols) > 0:
        max_vol = frontier_vols.max()
        cal_vols = np.linspace(0, max_vol * 1.2, 100)
        sharpe_max = portfolio_stats(max_sharpe_weights, annual_returns.values, cov_matrix.values, rf_rate)[2]
        cal_returns = rf_rate + sharpe_max * cal_vols
        
        fig.add_trace(go.Scatter(
            x=cal_vols * 100,
            y=cal_returns * 100,
            mode='lines',
            name='Capital Allocation Line',
            line=dict(color='gray', width=2, dash='dash'),
            hovertemplate='Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='Efficient Frontier with Optimized Portfolios',
        xaxis_title='Annual Volatility (%)',
        yaxis_title='Annual Return (%)',
        height=600,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.markdown("## 🎯 Risk Attribution")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Maximum Sharpe Portfolio")
    try:
        port_vol_ms = portfolio_stats(max_sharpe_weights, annual_returns.values, 
                                       cov_matrix.values, rf_rate)[1]
        marginal_contrib_ms = np.dot(cov_matrix.values, max_sharpe_weights)
        rc_ms = max_sharpe_weights * marginal_contrib_ms / (port_vol_ms + 1e-10)
        
        rc_df_ms = pd.DataFrame({
            'Asset': tickers,
            'Weight (%)': (max_sharpe_weights * 100).round(2),
            'Risk Contribution (%)': (rc_ms * 100).round(2)
        })
        
        fig_rc_ms = px.bar(rc_df_ms, x='Asset', y='Risk Contribution (%)',
                           color='Risk Contribution (%)',
                           color_continuous_scale='Blues')
        fig_rc_ms.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_rc_ms, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")

with col2:
    st.subheader("Risk Parity Portfolio")
    try:
        port_vol_rp = portfolio_stats(rp_weights, annual_returns.values, 
                                       cov_matrix.values, rf_rate)[1]
        marginal_contrib_rp = np.dot(cov_matrix.values, rp_weights)
        rc_rp = rp_weights * marginal_contrib_rp / (port_vol_rp + 1e-10)
        
        rc_df_rp = pd.DataFrame({
            'Asset': tickers,
            'Weight (%)': (rp_weights * 100).round(2),
            'Risk Contribution (%)': (rc_rp * 100).round(2)
        })
        
        fig_rc_rp = px.bar(rc_df_rp, x='Asset', y='Risk Contribution (%)',
                           color='Risk Contribution (%)',
                           color_continuous_scale='Greens')
        fig_rc_rp.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_rc_rp, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("## 🔗 Asset Correlations")

try:
    fig_cov = go.Figure(data=go.Heatmap(
        z=cov_matrix.values,
        x=tickers,
        y=tickers,
        colorscale='RdBu',
        zmid=0,
        text=np.round(cov_matrix.values, 3),
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Covariance")
    ))
    
    fig_cov.update_layout(
        title='Covariance Matrix (Annualized)',
        height=500
    )
    
    st.plotly_chart(fig_cov, use_container_width=True)
except Exception as e:
    st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("## 💾 Download Results")

try:
    download_data = pd.DataFrame({
        'Strategy': ['Minimum Variance', 'Maximum Sharpe', 'Risk Parity', 'Equal Weight'],
        'Annual Return (%)': results_df['Return (%)'].round(2),
        'Annual Volatility (%)': results_df['Volatility (%)'].round(2),
        'Sharpe Ratio': results_df['Sharpe Ratio'].round(3)
    })
    
    allocations_download = pd.concat([
        pd.DataFrame({'Strategy': 'Minimum Variance', 'Asset': tickers, 'Weight (%)': (min_var_weights*100).round(2)}),
        pd.DataFrame({'Strategy': 'Maximum Sharpe', 'Asset': tickers, 'Weight (%)': (max_sharpe_weights*100).round(2)}),
        pd.DataFrame({'Strategy': 'Risk Parity', 'Asset': tickers, 'Weight (%)': (rp_weights*100).round(2)}),
        pd.DataFrame({'Strategy': 'Equal Weight', 'Asset': tickers, 'Weight (%)': (ew_weights*100).round(2)})
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv1 = download_data.to_csv(index=False)
        st.download_button(
            label="📊 Download Performance Metrics",
            data=csv1,
            file_name="portfolio_performance.csv",
            mime="text/csv"
        )
    
    with col2:
        csv2 = allocations_download.to_csv(index=False)
        st.download_button(
            label="📈 Download Allocations",
            data=csv2,
            file_name="portfolio_allocations.csv",
            mime="text/csv"
        )
except Exception as e:
    st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("""
### 📚 About This Dashboard

**Institutional-grade portfolio optimization** engine implementing:
- Markowitz mean-variance framework
- Efficient frontier generation
- Maximum Sharpe ratio optimization
- Minimum variance allocation
- Risk parity methodology

Used by **BlackRock**, **Vanguard**, **Bridgewater**, **Citadel**.

---

**Built by:** Thu Nguyen | **Target:** Sydney Finance (IB/PE/Quant)  
**GitHub:** [github.com/thunguyen-debug](https://github.com/thunguyen-debug)  
**Email:** thunguyen5260@gmail.com | **LinkedIn:** [linkedin.com/in/thu-nguyen-00nvtt](https://linkedin.com/in/thu-nguyen-00nvtt)
""")

st.markdown("---")
st.success("✅ Dashboard loaded successfully!")        font-size: 2.5rem;
        color: #1f4788;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# GENERATE SAMPLE DATA (FALLBACK)
# ============================================================================

@st.cache_data
def generate_sample_data(tickers, days=1825):
    """Generate realistic sample data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Realistic parameters for different asset classes
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
        
        # Geometric Brownian Motion
        returns = np.random.normal(p['mu']/252, p['sigma']/np.sqrt(252), days)
        price = 100 * np.exp(np.cumsum(returns))
        data[ticker] = price
    
    return pd.DataFrame(data, index=dates)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.markdown("# ⚙️ Portfolio Settings")
st.sidebar.markdown("---")

# Data source selection
data_source = st.sidebar.radio(
    "📊 Data Source",
    ["Historical (Yahoo Finance)", "Sample Data (Demo)", "Upload CSV"],
    help="Use historical data from Yahoo Finance, sample data, or upload your own CSV"
)

if data_source == "Historical (Yahoo Finance)" or data_source == "Sample Data (Demo)":
    # Asset selection
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
    
    selected_assets = st.sidebar.multiselect(
        "🏢 Select Assets",
        list(available_assets.keys()),
        default=['US Equities', 'US Bonds', 'International Equities'],
        help="Choose which asset classes to include"
    )
    
    tickers = [available_assets[asset] for asset in selected_assets]
    
    # Time period selection
    time_period = st.sidebar.selectbox(
        "📅 Historical Period",
        ["1 Year", "3 Years", "5 Years", "10 Years"],
        index=2,
        help="How much historical data to use for analysis"
    )
    
    period_map = {
        "1 Year": 1 * 365,
        "3 Years": 3 * 365,
        "5 Years": 5 * 365,
        "10 Years": 10 * 365
    }
    
    days_back = period_map[time_period]
    
else:
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload CSV (returns data)",
        type=['csv'],
        help="CSV with date index and asset columns"
    )
    tickers = None

# Risk-free rate
st.sidebar.markdown("### 📈 Optimization Parameters")
rf_rate = st.sidebar.slider(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=4.5,
    step=0.1,
    help="Current 10-year US Treasury yield"
) / 100

# Covariance estimation method
cov_method = st.sidebar.selectbox(
    "Covariance Estimation",
    ["Sample", "Ledoit-Wolf Shrinkage"],
    help="Ledoit-Wolf is more robust, especially for small samples"
)

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

@st.cache_data
def load_data(tickers, days_back):
    """Download historical price data from Yahoo Finance"""
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
            
        except Exception as e:
            pass
    
    if len(data_list) == 0:
        return None, None
    
    data = pd.concat(data_list, axis=1)
    data.columns = successful_tickers
    data = data.dropna()
    
    return data, successful_tickers

@st.cache_data
def compute_statistics(data):
    """Compute returns, volatility, and covariance"""
    daily_returns = data.pct_change().dropna()
    annual_returns = daily_returns.mean() * 252
    annual_vol = daily_returns.std() * np.sqrt(252)
    cov_matrix = daily_returns.cov() * 252
    
    return daily_returns, annual_returns, annual_vol, cov_matrix

def estimate_covariance_robust(daily_returns):
    """Use Ledoit-Wolf shrinkage for robust covariance estimation"""
    try:
        lw = LedoitWolf()
        cov_shrink, _ = lw.fit(daily_returns)
        return pd.DataFrame(
            cov_shrink * 252,
            index=daily_returns.columns,
            columns=daily_returns.columns
        )
    except:
        return daily_returns.cov() * 252

# ============================================================================
# PORTFOLIO OPTIMIZATION FUNCTIONS
# ============================================================================

def portfolio_stats(weights, returns, cov_matrix, rf_rate):
    """Calculate portfolio statistics"""
    portfolio_return = np.sum(weights * returns)
    portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_vol = np.sqrt(portfolio_var)
    sharpe_ratio = (portfolio_return - rf_rate) / portfolio_vol if portfolio_vol > 0 else 0
    return portfolio_return, portfolio_vol, sharpe_ratio

def negative_sharpe(weights, returns, cov_matrix, rf_rate):
    """Objective function to maximize Sharpe ratio"""
    return -portfolio_stats(weights, returns, cov_matrix, rf_rate)[2]

def portfolio_volatility(weights, returns, cov_matrix, rf_rate):
    """Objective function to minimize volatility"""
    return portfolio_stats(weights, returns, cov_matrix, rf_rate)[1]

def optimize_portfolio(returns, cov_matrix, rf_rate, objective='sharpe'):
    """Run portfolio optimization"""
    n = len(returns)
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    
    if objective == 'sharpe':
        result = minimize(
            negative_sharpe,
            x0=np.array([1/n]*n),
            args=(returns, cov_matrix, rf_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
    elif objective == 'min_var':
        result = minimize(
            portfolio_volatility,
            x0=np.array([1/n]*n),
            args=(returns, cov_matrix, rf_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
    
    return result.x if result.success else None

def risk_parity_allocation(cov_matrix):
    """Compute risk-parity weights (equal risk contribution)"""
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
    """Generate efficient frontier"""
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
        
        result = minimize(
            portfolio_volatility,
            x0=np.array([1/n]*n),
            args=(returns, cov_matrix, rf_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            _, vol, _ = portfolio_stats(result.x, returns, cov_matrix, rf_rate)
            frontier_vols.append(vol)
            frontier_returns.append(target_ret)
    
    return np.array(frontier_returns), np.array(frontier_vols)

# ============================================================================
# MAIN APP
# ============================================================================

# Title
st.markdown('<div class="main-header">🎯 Portfolio Optimizer Dashboard</div>', 
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Institutional-grade mean-variance optimization & efficient frontier analysis</div>',
    unsafe_allow_html=True
)

# Load data
if data_source == "Historical (Yahoo Finance)":
    if not tickers:
        st.warning("⚠️ Please select at least one asset from the sidebar")
        st.stop()
    
    with st.spinner("📥 Loading historical data..."):
        try:
            data, successful_tickers = load_data(tickers, days_back)
            if data is None or len(data) == 0:
                st.warning("⚠️ Could not download from Yahoo Finance. Using sample data instead...")
                data = generate_sample_data(tickers, days_back)
                successful_tickers = tickers
            tickers = successful_tickers
        except Exception as e:
            st.warning("⚠️ Using sample data for demonstration...")
            data = generate_sample_data(tickers, days_back)
            
elif data_source == "Sample Data (Demo)":
    if not tickers:
        st.warning("⚠️ Please select at least one asset from the sidebar")
        st.stop()
    
    st.info("📊 Using realistic sample data for demonstration")
    data = generate_sample_data(tickers, days_back)
    
else:  # Upload CSV
    if uploaded_file is None:
        st.warning("⚠️ Please upload a CSV file")
        st.stop()
    
    try:
        data = pd.read_csv(uploaded_file, index_col=0)
        tickers = list(data.columns)
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {str(e)}")
        st.stop()

# Compute statistics
try:
    daily_returns, annual_returns, annual_vol, cov_matrix = compute_statistics(data)
except Exception as e:
    st.error(f"❌ Error computing statistics: {str(e)}")
    st.stop()

# Apply covariance method
if cov_method == "Ledoit-Wolf Shrinkage":
    try:
        cov_matrix = estimate_covariance_robust(daily_returns)
    except Exception as e:
        st.warning(f"⚠️ Using sample covariance instead")

# ============================================================================
# SECTION 1: ASSET SUMMARY
# ============================================================================

st.markdown("---")
st.markdown("## 📊 Asset Class Summary")

asset_summary = pd.DataFrame({
    'Asset': tickers,
    'Annual Return': (annual_returns.values * 100).round(2),
    'Annual Volatility': (annual_vol.values * 100).round(2),
    'Sharpe Ratio': ((annual_returns.values - rf_rate) / annual_vol.values).round(3)
})

col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(asset_summary, use_container_width=True, hide_index=True)
with col2:
    st.metric("Total Assets", len(tickers))
    st.metric("Analysis Period", f"{time_period}")
    st.metric("Risk-Free Rate", f"{rf_rate*100:.2f}%")

# ============================================================================
# SECTION 2: PORTFOLIO OPTIMIZATION
# ============================================================================

st.markdown("---")
st.markdown("## 🚀 Portfolio Optimization")

st.info("Computing optimal portfolios...")

try:
    min_var_weights = optimize_portfolio(annual_returns.values, cov_matrix.values, rf_rate, 'min_var')
    max_sharpe_weights = optimize_portfolio(annual_returns.values, cov_matrix.values, rf_rate, 'sharpe')
    rp_weights = risk_parity_allocation(cov_matrix.values)
    ew_weights = np.array([1.0/len(tickers)]*len(tickers))
    
    if min_var_weights is None or max_sharpe_weights is None:
        st.error("❌ Could not compute optimal portfolios.")
        st.stop()
    
except Exception as e:
    st.error(f"❌ Error in portfolio optimization: {str(e)}")
    st.stop()

# Compute statistics for each portfolio
def get_portfolio_stats(weights, name):
    ret, vol, sharpe = portfolio_stats(weights, annual_returns.values, cov_matrix.values, rf_rate)
    return {
        'Strategy': name,
        'Return (%)': ret * 100,
        'Volatility (%)': vol * 100,
        'Sharpe Ratio': sharpe,
        'Return/Risk': (ret / vol) if vol > 0 else 0
    }

results = [
    get_portfolio_stats(min_var_weights, 'Minimum Variance'),
    get_portfolio_stats(max_sharpe_weights, 'Maximum Sharpe'),
    get_portfolio_stats(rp_weights, 'Risk Parity'),
    get_portfolio_stats(ew_weights, 'Equal Weight')
]

results_df = pd.DataFrame(results)

# Display results
st.markdown("### Portfolio Optimization Results")
st.dataframe(results_df.set_index('Strategy'), use_container_width=True)

# ============================================================================
# SECTION 3: ALLOCATION COMPARISON
# ============================================================================

st.markdown("---")
st.markdown("## 📈 Portfolio Allocations")

col1, col2, col3, col4 = st.columns(4)

portfolios = [
    (min_var_weights, 'Minimum Variance', col1),
    (max_sharpe_weights, 'Maximum Sharpe', col2),
    (rp_weights, 'Risk Parity', col3),
    (ew_weights, 'Equal Weight', col4)
]

for weights, name, col in portfolios:
    with col:
        allocation = pd.DataFrame({
            'Asset': tickers,
            'Weight': (weights * 100).round(2)
        })
        allocation = allocation[allocation['Weight'] > 0.1]
        
        fig = go.Figure(data=[go.Pie(
            labels=allocation['Asset'],
            values=allocation['Weight'],
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>'
        )])
        fig.update_layout(
            title=name,
            height=400,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SECTION 4: EFFICIENT FRONTIER
# ============================================================================

st.markdown("---")
st.markdown("## 📊 Efficient Frontier Analysis")

st.info("Computing efficient frontier...")

try:
    frontier_returns, frontier_vols = compute_efficient_frontier(
        annual_returns.values, cov_matrix.values, rf_rate, n_points=80
    )
    
    # Create efficient frontier plot
    fig = go.Figure()
    
    # Add frontier
    fig.add_trace(go.Scatter(
        x=frontier_vols * 100,
        y=frontier_returns * 100,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='#1f4788', width=3),
        hovertemplate='Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
    ))
    
    # Add individual assets
    for i, ticker in enumerate(tickers):
        fig.add_trace(go.Scatter(
            x=[annual_vol.values[i] * 100],
            y=[annual_returns.values[i] * 100],
            mode='markers+text',
            name=ticker,
            marker=dict(size=10, opacity=0.7),
            text=[ticker],
            textposition='top center',
            hovertemplate=f'<b>{ticker}</b><br>Vol: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<extra></extra>'
        ))
    
    # Add optimized portfolios
    optimized = [
        (min_var_weights, 'Minimum Variance', '#FF6B6B'),
        (max_sharpe_weights, 'Maximum Sharpe', '#4ECDC4'),
        (rp_weights, 'Risk Parity', '#95E1D3'),
        (ew_weights, 'Equal Weight', '#FFE66D')
    ]
    
    for weights, name, color in optimized:
        ret, vol, sharpe = portfolio_stats(weights, annual_returns.values, cov_matrix.values, rf_rate)
        fig.add_trace(go.Scatter(
            x=[vol * 100],
            y=[ret * 100],
            mode='markers+text',
            name=name,
            marker=dict(size=15, symbol='star', color=color, line=dict(width=2, color='white')),
            text=[name],
            textposition='top center',
            hovertemplate=f'<b>{name}</b><br>Vol: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<br>Sharpe: {sharpe:.3f}<extra></extra>'
        ))
    
    # Add capital allocation line
    if len(frontier_vols) > 0:
        max_vol = frontier_vols.max()
        cal_vols = np.linspace(0, max_vol * 1.2, 100)
        sharpe_max = portfolio_stats(max_sharpe_weights, annual_returns.values, cov_matrix.values, rf_rate)[2]
        cal_returns = rf_rate + sharpe_max * cal_vols
        
        fig.add_trace(go.Scatter(
            x=cal_vols * 100,
            y=cal_returns * 100,
            mode='lines',
            name='Capital Allocation Line',
            line=dict(color='rgba(200, 200, 200, 0.5)', width=2, dash='dash'),
            hovertemplate='Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='Efficient Frontier with Optimized Portfolios',
        xaxis_title='Annual Volatility (%)',
        yaxis_title='Annual Return (%)',
        height=600,
        hovermode='closest',
        template='plotly_white',
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"❌ Error: {str(e)}")

# ============================================================================
# SECTION 5: RISK CONTRIBUTION ANALYSIS
# ============================================================================

st.markdown("---")
st.markdown("## 🎯 Risk Attribution")

col1, col2 = st.columns(2)

# Risk contribution for max sharpe
with col1:
    st.subheader("Maximum Sharpe Portfolio")
    try:
        port_vol_ms = portfolio_stats(max_sharpe_weights, annual_returns.values, 
                                       cov_matrix.values, rf_rate)[1]
        marginal_contrib_ms = np.dot(cov_matrix.values, max_sharpe_weights)
        rc_ms = max_sharpe_weights * marginal_contrib_ms / (port_vol_ms + 1e-10)
        
        rc_df_ms = pd.DataFrame({
            'Asset': tickers,
            'Weight (%)': (max_sharpe_weights * 100).round(2),
            'Risk Contribution (%)': (rc_ms * 100).round(2)
        })
        
        fig_rc_ms = px.bar(rc_df_ms, x='Asset', y='Risk Contribution (%)',
                           title='Risk Contribution by Asset',
                           color='Risk Contribution (%)',
                           color_continuous_scale='Blues')
        fig_rc_ms.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_rc_ms, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Risk contribution for risk parity
with col2:
    st.subheader("Risk Parity Portfolio")
    try:
        port_vol_rp = portfolio_stats(rp_weights, annual_returns.values, 
                                       cov_matrix.values, rf_rate)[1]
        marginal_contrib_rp = np.dot(cov_matrix.values, rp_weights)
        rc_rp = rp_weights * marginal_contrib_rp / (port_vol_rp + 1e-10)
        
        rc_df_rp = pd.DataFrame({
            'Asset': tickers,
            'Weight (%)': (rp_weights * 100).round(2),
            'Risk Contribution (%)': (rc_rp * 100).round(2)
        })
        
        fig_rc_rp = px.bar(rc_df_rp, x='Asset', y='Risk Contribution (%)',
                           title='Risk Contribution by Asset',
                           color='Risk Contribution (%)',
                           color_continuous_scale='Greens')
        fig_rc_rp.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_rc_rp, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")

# ============================================================================
# SECTION 6: COVARIANCE MATRIX HEATMAP
# ============================================================================

st.markdown("---")
st.markdown("## 🔗 Asset Correlations")

try:
    fig_cov = go.Figure(data=go.Heatmap(
        z=cov_matrix.values,
        x=tickers,
        y=tickers,
        colorscale='RdBu',
        zmid=0,
        text=np.round(cov_matrix.values, 3),
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Covariance")
    ))
    
    fig_cov.update_layout(
        title='Covariance Matrix (Annualized)',
        height=500
    )
    
    st.plotly_chart(fig_cov, use_container_width=True)
except Exception as e:
    st.error(f"Error: {str(e)}")

# ============================================================================
# SECTION 7: DOWNLOADABLE RESULTS
# ============================================================================

st.markdown("---")
st.markdown("## 💾 Download Results")

try:
    download_data = pd.DataFrame({
        'Strategy': ['Minimum Variance', 'Maximum Sharpe', 'Risk Parity', 'Equal Weight'],
        'Annual Return (%)': results_df['Return (%)'].round(2),
        'Annual Volatility (%)': results_df['Volatility (%)'].round(2),
        'Sharpe Ratio': results_df['Sharpe Ratio'].round(3)
    })
    
    allocations_download = pd.concat([
        pd.DataFrame({'Strategy': 'Minimum Variance', 'Asset': tickers, 'Weight (%)': (min_var_weights*100).round(2)}),
        pd.DataFrame({'Strategy': 'Maximum Sharpe', 'Asset': tickers, 'Weight (%)': (max_sharpe_weights*100).round(2)}),
        pd.DataFrame({'Strategy': 'Risk Parity', 'Asset': tickers, 'Weight (%)': (rp_weights*100).round(2)}),
        pd.DataFrame({'Strategy': 'Equal Weight', 'Asset': tickers, 'Weight (%)': (ew_weights*100).round(2)})
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv1 = download_data.to_csv(index=False)
        st.download_button(
            label="📊 Download Performance Metrics (CSV)",
            data=csv1,
            file_name="portfolio_performance.csv",
            mime="text/csv"
        )
    
    with col2:
        csv2 = allocations_download.to_csv(index=False)
        st.download_button(
            label="📈 Download Allocations (CSV)",
            data=csv2,
            file_name="portfolio_allocations.csv",
            mime="text/csv"
        )
except Exception as e:
    st.error(f"Error: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
### 📚 About This Dashboard

This dashboard implements **institutional-grade portfolio optimization** used by:
- Asset managers (BlackRock, Vanguard, State Street)
- Hedge funds (Bridgewater, Citadel, Two Sigma)
- Pension funds and endowments

**Key Concepts:**
- **Efficient Frontier**: Set of portfolios that minimize risk for each return level
- **Maximum Sharpe**: Portfolio with best risk-adjusted returns
- **Minimum Variance**: Portfolio with lowest volatility
- **Risk Parity**: Portfolio where each asset contributes equally to risk
- **Capital Allocation Line**: Shows risk-return tradeoff

**Data Source**: Yahoo Finance (or sample data if unavailable)  
**Time Period**: Configurable (1Y, 3Y, 5Y, 10Y)  

---

**Built for:** Finance recruiting (IB, PE, Quant)  
**GitHub**: [github.com/thunguyen-debug/portfolio-optimization-mpt](https://github.com/thunguyen-debug/portfolio-optimization-mpt)  
**Author**: Thu Nguyen | Targeting Sydney finance roles

""")

st.markdown("---")
st.success("✅ Dashboard loaded successfully!")
css_code = """
    :root {
        --primary: #00d4ff;
        --secondary: #ff006e;
        --accent: #ffd60a;
        --dark-bg: #0a0e27;
        --card-bg: #1a1f3a;
        --border: #2a3055;
        --text-primary: #ffffff;
        --text-secondary: #a0aec0;
        --success: #10b981;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1429 0%, #1a1f3a 100%);
        border-right: 1px solid #2a3055;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    h2 {
        font-size: 1.75rem;
        margin: 2rem 0 1rem 0;
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff 0%, #ff006e 50%, #ffd60a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #a0aec0;
        margin-bottom: 2rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricContainer"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.05) 0%, rgba(255, 0, 110, 0.05) 100%);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stDataFrame"] {
        background: linear-gradient(135deg, rgba(26, 31, 58, 0.8) 0%, rgba(42, 48, 85, 0.8) 100%);
        border: 1px solid #2a3055;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .dataframe {
        background: transparent !important;
        color: #ffffff !important;
    }
    
    .dataframe tbody tr:nth-child(odd) {
        background: rgba(0, 212, 255, 0.03) !important;
    }
    
    .dataframe tbody tr:hover {
        background: rgba(0, 212, 255, 0.1) !important;
    }
    
    .dataframe th {
        background: rgba(0, 212, 255, 0.1) !important;
        color: #00d4ff !important;
        border-color: #2a3055 !important;
        font-weight: 600;
    }
    
    .dataframe td {
        border-color: #2a3055 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stAlert"] {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(0, 212, 255, 0.1) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        color: #10b981;
    }
    
    input, select, textarea {
        background: rgba(26, 31, 58, 0.6) !important;
        border: 1px solid #2a3055 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
    }
    
    input:focus, select:focus, textarea:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.2) !important;
    }
    
    [data-testid="stButton"] button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        border: none;
        color: #000000;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }
    
    [data-testid="stButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 212, 255, 0.5);
    }
    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #2a3055, transparent);
        margin: 2rem 0;
    }
    
    [data-testid="stMetricValue"] {
        color: #00d4ff;
        font-size: 2rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0aec0;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(26, 31, 58, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: #00d4ff;
        border-radius: 4px;
    }
"""

st.markdown(f"<style>{css_code}</style>", unsafe_allow_html=True)

# ============================================================================
# GENERATE SAMPLE DATA
# ============================================================================

@st.cache_data
def generate_sample_data(tickers, days=1825):
    """Generate realistic sample data for demonstration"""
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

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.markdown("# ⚙️ Portfolio Settings")
st.sidebar.markdown("---")

data_source = st.sidebar.radio(
    "📊 Data Source",
    ["Historical (Yahoo Finance)", "Sample Data (Demo)", "Upload CSV"],
    help="Use historical data from Yahoo Finance, sample data, or upload your own CSV"
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
    
    selected_assets = st.sidebar.multiselect(
        "🏢 Select Assets",
        list(available_assets.keys()),
        default=['US Equities', 'US Bonds', 'International Equities'],
        help="Choose which asset classes to include"
    )
    
    tickers = [available_assets[asset] for asset in selected_assets]
    
    time_period = st.sidebar.selectbox(
        "📅 Historical Period",
        ["1 Year", "3 Years", "5 Years", "10 Years"],
        index=2,
        help="How much historical data to use for analysis"
    )
    
    period_map = {
        "1 Year": 1 * 365,
        "3 Years": 3 * 365,
        "5 Years": 5 * 365,
        "10 Years": 10 * 365
    }
    
    days_back = period_map[time_period]
    
else:
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload CSV (returns data)",
        type=['csv'],
        help="CSV with date index and asset columns"
    )
    tickers = None

st.sidebar.markdown("### 📈 Optimization Parameters")
rf_rate = st.sidebar.slider(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=4.5,
    step=0.1,
    help="Current 10-year US Treasury yield"
) / 100

cov_method = st.sidebar.selectbox(
    "Covariance Estimation",
    ["Sample", "Ledoit-Wolf Shrinkage"],
    help="Ledoit-Wolf is more robust, especially for small samples"
)

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

@st.cache_data
def load_data(tickers, days_back):
    """Download historical price data from Yahoo Finance"""
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
        except Exception as e:
            pass
    
    if len(data_list) == 0:
        return None, None
    
    data = pd.concat(data_list, axis=1)
    data.columns = successful_tickers
    data = data.dropna()
    
    return data, successful_tickers

@st.cache_data
def compute_statistics(data):
    """Compute returns, volatility, and covariance"""
    daily_returns = data.pct_change().dropna()
    annual_returns = daily_returns.mean() * 252
    annual_vol = daily_returns.std() * np.sqrt(252)
    cov_matrix = daily_returns.cov() * 252
    
    return daily_returns, annual_returns, annual_vol, cov_matrix

def estimate_covariance_robust(daily_returns):
    """Use Ledoit-Wolf shrinkage for robust covariance estimation"""
    try:
        lw = LedoitWolf()
        cov_shrink, _ = lw.fit(daily_returns)
        return pd.DataFrame(
            cov_shrink * 252,
            index=daily_returns.columns,
            columns=daily_returns.columns
        )
    except:
        return daily_returns.cov() * 252

# ============================================================================
# PORTFOLIO OPTIMIZATION FUNCTIONS
# ============================================================================

def portfolio_stats(weights, returns, cov_matrix, rf_rate):
    """Calculate portfolio statistics"""
    portfolio_return = np.sum(weights * returns)
    portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_vol = np.sqrt(portfolio_var)
    sharpe_ratio = (portfolio_return - rf_rate) / portfolio_vol if portfolio_vol > 0 else 0
    return portfolio_return, portfolio_vol, sharpe_ratio

def negative_sharpe(weights, returns, cov_matrix, rf_rate):
    """Objective function to maximize Sharpe ratio"""
    return -portfolio_stats(weights, returns, cov_matrix, rf_rate)[2]

def portfolio_volatility(weights, returns, cov_matrix, rf_rate):
    """Objective function to minimize volatility"""
    return portfolio_stats(weights, returns, cov_matrix, rf_rate)[1]

def optimize_portfolio(returns, cov_matrix, rf_rate, objective='sharpe'):
    """Run portfolio optimization"""
    n = len(returns)
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    
    if objective == 'sharpe':
        result = minimize(
            negative_sharpe,
            x0=np.array([1/n]*n),
            args=(returns, cov_matrix, rf_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
    elif objective == 'min_var':
        result = minimize(
            portfolio_volatility,
            x0=np.array([1/n]*n),
            args=(returns, cov_matrix, rf_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
    
    return result.x if result.success else None

def risk_parity_allocation(cov_matrix):
    """Compute risk-parity weights (equal risk contribution)"""
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
    """Generate efficient frontier"""
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
        
        result = minimize(
            portfolio_volatility,
            x0=np.array([1/n]*n),
            args=(returns, cov_matrix, rf_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            _, vol, _ = portfolio_stats(result.x, returns, cov_matrix, rf_rate)
            frontier_vols.append(vol)
            frontier_returns.append(target_ret)
    
    return np.array(frontier_returns), np.array(frontier_vols)

# ============================================================================
# MAIN APP
# ============================================================================

st.markdown('<div class="main-header">🎯 Portfolio Optimizer</div>', 
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Institutional-grade mean-variance optimization & efficient frontier analysis</div>',
    unsafe_allow_html=True
)

# Load data
if data_source == "Historical (Yahoo Finance)":
    if not tickers:
        st.warning("⚠️ Please select at least one asset from the sidebar")
        st.stop()
    
    with st.spinner("📥 Loading historical data..."):
        try:
            data, successful_tickers = load_data(tickers, days_back)
            if data is None or len(data) == 0:
                st.warning("⚠️ Could not download from Yahoo Finance. Using sample data instead...")
                data = generate_sample_data(tickers, days_back)
                successful_tickers = tickers
            tickers = successful_tickers
        except Exception as e:
            st.warning("⚠️ Using sample data for demonstration...")
            data = generate_sample_data(tickers, days_back)
            
elif data_source == "Sample Data (Demo)":
    if not tickers:
        st.warning("⚠️ Please select at least one asset from the sidebar")
        st.stop()
    
    st.info("📊 Using realistic sample data for demonstration")
    data = generate_sample_data(tickers, days_back)
    
else:
    if uploaded_file is None:
        st.warning("⚠️ Please upload a CSV file")
        st.stop()
    
    try:
        data = pd.read_csv(uploaded_file, index_col=0)
        tickers = list(data.columns)
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {str(e)}")
        st.stop()

# Compute statistics
try:
    daily_returns, annual_returns, annual_vol, cov_matrix = compute_statistics(data)
except Exception as e:
    st.error(f"❌ Error computing statistics: {str(e)}")
    st.stop()

if cov_method == "Ledoit-Wolf Shrinkage":
    try:
        cov_matrix = estimate_covariance_robust(daily_returns)
    except Exception as e:
        st.warning(f"⚠️ Using sample covariance instead")

# ============================================================================
# SECTION 1: ASSET SUMMARY
# ============================================================================

st.markdown("---")
st.markdown("## 📊 Asset Class Summary")

asset_summary = pd.DataFrame({
    'Asset': tickers,
    'Annual Return': (annual_returns.values * 100).round(2),
    'Annual Volatility': (annual_vol.values * 100).round(2),
    'Sharpe Ratio': ((annual_returns.values - rf_rate) / annual_vol.values).round(3)
})

col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(asset_summary, use_container_width=True, hide_index=True)
with col2:
    st.metric("Total Assets", len(tickers))
    st.metric("Analysis Period", f"{time_period}")
    st.metric("Risk-Free Rate", f"{rf_rate*100:.2f}%")

# ============================================================================
# SECTION 2: PORTFOLIO OPTIMIZATION
# ============================================================================

st.markdown("---")
st.markdown("## 🚀 Portfolio Optimization")

st.info("Computing optimal portfolios...")

try:
    min_var_weights = optimize_portfolio(annual_returns.values, cov_matrix.values, rf_rate, 'min_var')
    max_sharpe_weights = optimize_portfolio(annual_returns.values, cov_matrix.values, rf_rate, 'sharpe')
    rp_weights = risk_parity_allocation(cov_matrix.values)
    ew_weights = np.array([1.0/len(tickers)]*len(tickers))
    
    if min_var_weights is None or max_sharpe_weights is None:
        st.error("❌ Could not compute optimal portfolios.")
        st.stop()
    
except Exception as e:
    st.error(f"❌ Error in portfolio optimization: {str(e)}")
    st.stop()

def get_portfolio_stats(weights, name):
    ret, vol, sharpe = portfolio_stats(weights, annual_returns.values, cov_matrix.values, rf_rate)
    return {
        'Strategy': name,
        'Return (%)': ret * 100,
        'Volatility (%)': vol * 100,
        'Sharpe Ratio': sharpe,
        'Return/Risk': (ret / vol) if vol > 0 else 0
    }

results = [
    get_portfolio_stats(min_var_weights, 'Minimum Variance'),
    get_portfolio_stats(max_sharpe_weights, 'Maximum Sharpe'),
    get_portfolio_stats(rp_weights, 'Risk Parity'),
    get_portfolio_stats(ew_weights, 'Equal Weight')
]

results_df = pd.DataFrame(results)

st.markdown("### Portfolio Optimization Results")
st.dataframe(results_df.set_index('Strategy'), use_container_width=True)

# ============================================================================
# SECTION 3: ALLOCATION COMPARISON
# ============================================================================

st.markdown("---")
st.markdown("## 📈 Portfolio Allocations")

col1, col2, col3, col4 = st.columns(4)

portfolios = [
    (min_var_weights, 'Minimum Variance', col1),
    (max_sharpe_weights, 'Maximum Sharpe', col2),
    (rp_weights, 'Risk Parity', col3),
    (ew_weights, 'Equal Weight', col4)
]

for weights, name, col in portfolios:
    with col:
        allocation = pd.DataFrame({
            'Asset': tickers,
            'Weight': (weights * 100).round(2)
        })
        allocation = allocation[allocation['Weight'] > 0.1]
        
        fig = go.Figure(data=[go.Pie(
            labels=allocation['Asset'],
            values=allocation['Weight'],
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>'
        )])
        fig.update_layout(
            title=name,
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff')
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ============================================================================
# SECTION 4: EFFICIENT FRONTIER
# ============================================================================

st.markdown("---")
st.markdown("## 📊 Efficient Frontier Analysis")

st.info("Computing efficient frontier...")

try:
    frontier_returns, frontier_vols = compute_efficient_frontier(
        annual_returns.values, cov_matrix.values, rf_rate, n_points=80
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=frontier_vols * 100,
        y=frontier_returns * 100,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='#00d4ff', width=4),
        hovertemplate='Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
    ))
    
    for i, ticker in enumerate(tickers):
        fig.add_trace(go.Scatter(
            x=[annual_vol.values[i] * 100],
            y=[annual_returns.values[i] * 100],
            mode='markers+text',
            name=ticker,
            marker=dict(size=12, opacity=0.8, color='#ffd60a', line=dict(width=2, color='#ffffff')),
            text=[ticker],
            textposition='top center',
            textfont=dict(color='#ffffff', size=10),
            hovertemplate=f'<b>{ticker}</b><br>Vol: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<extra></extra>'
        ))
    
    optimized = [
        (min_var_weights, 'Min Variance', '#10b981'),
        (max_sharpe_weights, 'Max Sharpe', '#ff006e'),
        (rp_weights, 'Risk Parity', '#00d4ff'),
        (ew_weights, 'Equal Weight', '#ffd60a')
    ]
    
    for weights, name, color in optimized:
        ret, vol, sharpe = portfolio_stats(weights, annual_returns.values, cov_matrix.values, rf_rate)
        fig.add_trace(go.Scatter(
            x=[vol * 100],
            y=[ret * 100],
            mode='markers',
            name=name,
            marker=dict(size=20, symbol='star', color=color, line=dict(width=3, color='#ffffff')),
            hovertemplate=f'<b>{name}</b><br>Vol: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<br>Sharpe: {sharpe:.3f}<extra></extra>'
        ))
    
    if len(frontier_vols) > 0:
        max_vol = frontier_vols.max()
        cal_vols = np.linspace(0, max_vol * 1.2, 100)
        sharpe_max = portfolio_stats(max_sharpe_weights, annual_returns.values, cov_matrix.values, rf_rate)[2]
        cal_returns = rf_rate + sharpe_max * cal_vols
        
        fig.add_trace(go.Scatter(
            x=cal_vols * 100,
            y=cal_returns * 100,
            mode='lines',
            name='Capital Allocation Line',
            line=dict(color='rgba(160, 174, 192, 0.5)', width=2, dash='dash'),
            hovertemplate='Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='Efficient Frontier with Optimized Portfolios',
        xaxis_title='Annual Volatility (%)',
        yaxis_title='Annual Return (%)',
        height=600,
        hovermode='closest',
        template='plotly_dark',
        paper_bgcolor='rgba(10, 14, 39, 0.5)',
        plot_bgcolor='rgba(26, 31, 58, 0.3)',
        font=dict(color='#ffffff', size=12),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"❌ Error: {str(e)}")

# ============================================================================
# SECTION 5: RISK CONTRIBUTION ANALYSIS
# ============================================================================

st.markdown("---")
st.markdown("## 🎯 Risk Attribution")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Maximum Sharpe Portfolio")
    try:
        port_vol_ms = portfolio_stats(max_sharpe_weights, annual_returns.values, 
                                       cov_matrix.values, rf_rate)[1]
        marginal_contrib_ms = np.dot(cov_matrix.values, max_sharpe_weights)
        rc_ms = max_sharpe_weights * marginal_contrib_ms / (port_vol_ms + 1e-10)
        
        rc_df_ms = pd.DataFrame({
            'Asset': tickers,
            'Weight (%)': (max_sharpe_weights * 100).round(2),
            'Risk Contribution (%)': (rc_ms * 100).round(2)
        })
        
        fig_rc_ms = px.bar(rc_df_ms, x='Asset', y='Risk Contribution (%)',
                           color='Risk Contribution (%)',
                           color_continuous_scale=[[0, '#00d4ff'], [1, '#0099cc']])
        fig_rc_ms.update_layout(height=400, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#ffffff'))
        st.plotly_chart(fig_rc_ms, use_container_width=True, config={'displayModeBar': False})
    except Exception as e:
        st.error(f"Error: {str(e)}")

with col2:
    st.subheader("Risk Parity Portfolio")
    try:
        port_vol_rp = portfolio_stats(rp_weights, annual_returns.values, 
                                       cov_matrix.values, rf_rate)[1]
        marginal_contrib_rp = np.dot(cov_matrix.values, rp_weights)
        rc_rp = rp_weights * marginal_contrib_rp / (port_vol_rp + 1e-10)
        
        rc_df_rp = pd.DataFrame({
            'Asset': tickers,
            'Weight (%)': (rp_weights * 100).round(2),
            'Risk Contribution (%)': (rc_rp * 100).round(2)
        })
        
        fig_rc_rp = px.bar(rc_df_rp, x='Asset', y='Risk Contribution (%)',
                           color='Risk Contribution (%)',
                           color_continuous_scale=[[0, '#10b981'], [1, '#059669']])
        fig_rc_rp.update_layout(height=400, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#ffffff'))
        st.plotly_chart(fig_rc_rp, use_container_width=True, config={'displayModeBar': False})
    except Exception as e:
        st.error(f"Error: {str(e)}")

# ============================================================================
# SECTION 6: COVARIANCE MATRIX HEATMAP
# ============================================================================

st.markdown("---")
st.markdown("## 🔗 Asset Correlations")

try:
    fig_cov = go.Figure(data=go.Heatmap(
        z=cov_matrix.values,
        x=tickers,
        y=tickers,
        colorscale='RdBu',
        zmid=0,
        text=np.round(cov_matrix.values, 3),
        texttemplate='%{text:.2f}',
        textfont={"size": 11}
    ))
    
    fig_cov.update_layout(
        title='Covariance Matrix (Annualized)',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff')
    )
    
    st.plotly_chart(fig_cov, use_container_width=True)
except Exception as e:
    st.error(f"Error: {str(e)}")

# ============================================================================
# SECTION 7: DOWNLOADABLE RESULTS
# ============================================================================

st.markdown("---")
st.markdown("## 💾 Download Results")

try:
    download_data = pd.DataFrame({
        'Strategy': ['Minimum Variance', 'Maximum Sharpe', 'Risk Parity', 'Equal Weight'],
        'Annual Return (%)': results_df['Return (%)'].round(2),
        'Annual Volatility (%)': results_df['Volatility (%)'].round(2),
        'Sharpe Ratio': results_df['Sharpe Ratio'].round(3)
    })
    
    allocations_download = pd.concat([
        pd.DataFrame({'Strategy': 'Minimum Variance', 'Asset': tickers, 'Weight (%)': (min_var_weights*100).round(2)}),
        pd.DataFrame({'Strategy': 'Maximum Sharpe', 'Asset': tickers, 'Weight (%)': (max_sharpe_weights*100).round(2)}),
        pd.DataFrame({'Strategy': 'Risk Parity', 'Asset': tickers, 'Weight (%)': (rp_weights*100).round(2)}),
        pd.DataFrame({'Strategy': 'Equal Weight', 'Asset': tickers, 'Weight (%)': (ew_weights*100).round(2)})
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv1 = download_data.to_csv(index=False)
        st.download_button(
            label="📊 Download Performance Metrics",
            data=csv1,
            file_name="portfolio_performance.csv",
            mime="text/csv"
        )
    
    with col2:
        csv2 = allocations_download.to_csv(index=False)
        st.download_button(
            label="📈 Download Allocations",
            data=csv2,
            file_name="portfolio_allocations.csv",
            mime="text/csv"
        )
except Exception as e:
    st.error(f"Error: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
### 📚 About This Dashboard

**Institutional-grade portfolio optimization** engine implementing:
- Markowitz mean-variance framework
- Efficient frontier generation
- Maximum Sharpe ratio optimization
- Minimum variance allocation
- Risk parity methodology

Used by **BlackRock**, **Vanguard**, **Bridgewater**, **Citadel** and other tier-1 institutions.

---

**Built by:** Thu Nguyen | **Target:** Sydney Finance (IB/PE/Quant)  
**GitHub:** [github.com/thunguyen-debug](https://github.com/thunguyen-debug)  
**Email:** thunguyen5260@gmail.com | **LinkedIn:** [linkedin.com/in/thu-nguyen-00nvtt](https://linkedin.com/in/thu-nguyen-00nvtt)

""")

st.markdown("---")
st.success("✅ Dashboard loaded successfully!")
# ============================================================================

st.markdown("""
<style>
    /* Root Variables */
    :root {
        --primary: #00d4ff;
        --secondary: #ff006e;
        --accent: #ffd60a;
        --dark-bg: #0a0e27;
        --card-bg: #1a1f3a;
        --border: #2a3055;
        --text-primary: #ffffff;
        --text-secondary: #a0aec0;
        --success: #10b981;
    }

    /* Global Styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Main Container */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1429 0%, #1a1f3a 100%);
        border-right: 1px solid #2a3055;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        padding-left: 1.5rem;
    }

    /* Main Content */
    .main {
        background: transparent;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-weight: 600;
        letter-spacing: -0.5px;
    }

    h2 {
        font-size: 1.75rem;
        margin: 2rem 0 1rem 0;
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Main Header */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff 0%, #ff006e 50%, #ffd60a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }

    /* Sub Header */
    .sub-header {
        font-size: 1.1rem;
        color: #a0aec0;
        margin-bottom: 2rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }

    /* Cards & Containers */
    [data-testid="stMetricContainer"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.05) 0%, rgba(255, 0, 110, 0.05) 100%);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
    }

    /* Dataframe Styles */
    [data-testid="stDataFrame"] {
        background: linear-gradient(135deg, rgba(26, 31, 58, 0.8) 0%, rgba(42, 48, 85, 0.8) 100%);
        border: 1px solid #2a3055;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }

    .dataframe {
        background: transparent !important;
        color: #ffffff !important;
    }

    .dataframe tbody tr:nth-child(odd) {
        background: rgba(0, 212, 255, 0.03) !important;
    }

    .dataframe tbody tr:hover {
        background: rgba(0, 212, 255, 0.1) !important;
    }

    .dataframe th {
        background: rgba(0, 212, 255, 0.1) !important;
        color: #00d4ff !important;
        border-color: #2a3055 !important;
        font-weight: 600;
    }

    .dataframe td {
        border-color: #2a3055 !important;
        color: #ffffff !important;
    }

    /* Info Boxes */
    [data-testid="stAlert"] {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(0, 212, 255, 0.1) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        color: #10b981;
    }

    /* Warning Boxes */
    [role="alert"] {
        background: linear-gradient(135deg, rgba(255, 214, 10, 0.1) 0%, rgba(255, 0, 110, 0.1) 100%);
        border: 1px solid rgba(255, 214, 10, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }

    /* Input Elements */
    input, select, textarea {
        background: rgba(26, 31, 58, 0.6) !important;
        border: 1px solid #2a3055 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
    }

    input:focus, select:focus, textarea:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.2) !important;
    }

    /* Buttons */
    [data-testid="stButton"] button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        border: none;
        color: #000000;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }

    [data-testid="stButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 212, 255, 0.5);
    }

    /* Selectbox */
    [data-baseweb="select"] {
        background: rgba(26, 31, 58, 0.6) !important;
        border-radius: 8px !important;
    }

    /* Slider */
    [data-testid="stSlider"] {
        padding: 1rem 0;
    }

    .stSlider [data-testid="stThumb"] {
        background: #00d4ff;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #2a3055, transparent);
        margin: 2rem 0;
    }

    /* Metric Value */
    [data-testid="stMetricValue"] {
        color: #00d4ff;
        font-size: 2rem;
        font-weight: 700;
    }

    /* Metric Label */
    [data-testid="stMetricLabel"] {
        color: #a0aec0;
        font-size: 0.875rem;
        font-weight: 500;
    }

    /* Plotly Charts */
    .plotly-graph-div {
        background: transparent !important;
    }

    /* Sidebar Title */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2 {
        color: #00d4ff;
        font-size: 1.25rem;
        margin-top: 0;
    }

    /* Radio Buttons */
    [data-testid="stRadio"] {
        padding: 0.5rem 0;
    }

    [data-testid="stRadio"] [role="radio"] {
        accent-color: #00d4ff;
    }

    /* Multiselect */
    [data-testid="stMultiSelect"] [data-testid="stMarkdownContainer"] {
        color: #a0aec0;
    }

    /* Tags/Pills */
    [data-testid="stMultiSelect"] [data-testid="stMarkdownContainer"] span {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: #000000 !important;
        border-radius: 6px;
        padding: 0.25rem 0.75rem;
        font-weight: 600;
        margin: 0.25rem;
        display: inline-block;
    }

    /* Spinner */
    .stSpinner {
        color: #00d4ff;
    }

    /* Text */
    p {
        color: #a0aec0;
        line-height: 1.6;
    }

    /* Links */
    a {
        color: #00d4ff;
        text-decoration: none;
        transition: all 0.3s ease;
    }

    a:hover {
        color: #ff006e;
        text-decoration: underline;
    }

    /* Success Message */
    [data-testid="stAlert"][role="status"] {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(0, 212, 255, 0.15) 100%);
        border: 1px solid #10b981;
        color: #10b981;
    }

    /* Footer */
    footer {
        background: transparent;
        border-top: 1px solid #2a3055;
        color: #a0aec0;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(26, 31, 58, 0.5);
    }

    ::-webkit-scrollbar-thumb {
        background: #00d4ff;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #0099cc;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# GENERATE SAMPLE DATA (FALLBACK)
# ============================================================================

@st.cache_data
def generate_sample_data(tickers, days=1825):
    """Generate realistic sample data for demonstration"""
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

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.markdown("# ⚙️ Portfolio Settings")
st.sidebar.markdown("---")

data_source = st.sidebar.radio(
    "📊 Data Source",
    ["Historical (Yahoo Finance)", "Sample Data (Demo)", "Upload CSV"],
    help="Use historical data from Yahoo Finance, sample data, or upload your own CSV"
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
    
    selected_assets = st.sidebar.multiselect(
        "🏢 Select Assets",
        list(available_assets.keys()),
        default=['US Equities', 'US Bonds', 'International Equities'],
        help="Choose which asset classes to include"
    )
    
    tickers = [available_assets[asset] for asset in selected_assets]
    
    time_period = st.sidebar.selectbox(
        "📅 Historical Period",
        ["1 Year", "3 Years", "5 Years", "10 Years"],
        index=2,
        help="How much historical data to use for analysis"
    )
    
    period_map = {
        "1 Year": 1 * 365,
        "3 Years": 3 * 365,
        "5 Years": 5 * 365,
        "10 Years": 10 * 365
    }
    
    days_back = period_map[time_period]
    
else:
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload CSV (returns data)",
        type=['csv'],
        help="CSV with date index and asset columns"
    )
    tickers = None

st.sidebar.markdown("### 📈 Optimization Parameters")
rf_rate = st.sidebar.slider(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=4.5,
    step=0.1,
    help="Current 10-year US Treasury yield"
) / 100

cov_method = st.sidebar.selectbox(
    "Covariance Estimation",
    ["Sample", "Ledoit-Wolf Shrinkage"],
    help="Ledoit-Wolf is more robust, especially for small samples"
)

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

@st.cache_data
def load_data(tickers, days_back):
    """Download historical price data from Yahoo Finance"""
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
        except Exception as e:
            pass
    
    if len(data_list) == 0:
        return None, None
    
    data = pd.concat(data_list, axis=1)
    data.columns = successful_tickers
    data = data.dropna()
    
    return data, successful_tickers

@st.cache_data
def compute_statistics(data):
    """Compute returns, volatility, and covariance"""
    daily_returns = data.pct_change().dropna()
    annual_returns = daily_returns.mean() * 252
    annual_vol = daily_returns.std() * np.sqrt(252)
    cov_matrix = daily_returns.cov() * 252
    
    return daily_returns, annual_returns, annual_vol, cov_matrix

def estimate_covariance_robust(daily_returns):
    """Use Ledoit-Wolf shrinkage for robust covariance estimation"""
    try:
        lw = LedoitWolf()
        cov_shrink, _ = lw.fit(daily_returns)
        return pd.DataFrame(
            cov_shrink * 252,
            index=daily_returns.columns,
            columns=daily_returns.columns
        )
    except:
        return daily_returns.cov() * 252

# ============================================================================
# PORTFOLIO OPTIMIZATION FUNCTIONS
# ============================================================================

def portfolio_stats(weights, returns, cov_matrix, rf_rate):
    """Calculate portfolio statistics"""
    portfolio_return = np.sum(weights * returns)
    portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_vol = np.sqrt(portfolio_var)
    sharpe_ratio = (portfolio_return - rf_rate) / portfolio_vol if portfolio_vol > 0 else 0
    return portfolio_return, portfolio_vol, sharpe_ratio

def negative_sharpe(weights, returns, cov_matrix, rf_rate):
    """Objective function to maximize Sharpe ratio"""
    return -portfolio_stats(weights, returns, cov_matrix, rf_rate)[2]

def portfolio_volatility(weights, returns, cov_matrix, rf_rate):
    """Objective function to minimize volatility"""
    return portfolio_stats(weights, returns, cov_matrix, rf_rate)[1]

def optimize_portfolio(returns, cov_matrix, rf_rate, objective='sharpe'):
    """Run portfolio optimization"""
    n = len(returns)
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    
    if objective == 'sharpe':
        result = minimize(
            negative_sharpe,
            x0=np.array([1/n]*n),
            args=(returns, cov_matrix, rf_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
    elif objective == 'min_var':
        result = minimize(
            portfolio_volatility,
            x0=np.array([1/n]*n),
            args=(returns, cov_matrix, rf_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
    
    return result.x if result.success else None

def risk_parity_allocation(cov_matrix):
    """Compute risk-parity weights (equal risk contribution)"""
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
    """Generate efficient frontier"""
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
        
        result = minimize(
            portfolio_volatility,
            x0=np.array([1/n]*n),
            args=(returns, cov_matrix, rf_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            _, vol, _ = portfolio_stats(result.x, returns, cov_matrix, rf_rate)
            frontier_vols.append(vol)
            frontier_returns.append(target_ret)
    
    return np.array(frontier_returns), np.array(frontier_vols)

# ============================================================================
# MAIN APP
# ============================================================================

st.markdown('<div class="main-header">🎯 Portfolio Optimizer</div>', 
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Institutional-grade mean-variance optimization & efficient frontier analysis</div>',
    unsafe_allow_html=True
)

# Load data
if data_source == "Historical (Yahoo Finance)":
    if not tickers:
        st.warning("⚠️ Please select at least one asset from the sidebar")
        st.stop()
    
    with st.spinner("📥 Loading historical data..."):
        try:
            data, successful_tickers = load_data(tickers, days_back)
            if data is None or len(data) == 0:
                st.warning("⚠️ Could not download from Yahoo Finance. Using sample data instead...")
                data = generate_sample_data(tickers, days_back)
                successful_tickers = tickers
            tickers = successful_tickers
        except Exception as e:
            st.warning("⚠️ Using sample data for demonstration...")
            data = generate_sample_data(tickers, days_back)
            
elif data_source == "Sample Data (Demo)":
    if not tickers:
        st.warning("⚠️ Please select at least one asset from the sidebar")
        st.stop()
    
    st.info("📊 Using realistic sample data for demonstration")
    data = generate_sample_data(tickers, days_back)
    
else:
    if uploaded_file is None:
        st.warning("⚠️ Please upload a CSV file")
        st.stop()
    
    try:
        data = pd.read_csv(uploaded_file, index_col=0)
        tickers = list(data.columns)
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {str(e)}")
        st.stop()

# Compute statistics
try:
    daily_returns, annual_returns, annual_vol, cov_matrix = compute_statistics(data)
except Exception as e:
    st.error(f"❌ Error computing statistics: {str(e)}")
    st.stop()

if cov_method == "Ledoit-Wolf Shrinkage":
    try:
        cov_matrix = estimate_covariance_robust(daily_returns)
    except Exception as e:
        st.warning(f"⚠️ Using sample covariance instead")

# ============================================================================
# SECTION 1: ASSET SUMMARY
# ============================================================================

st.markdown("---")
st.markdown("## 📊 Asset Class Summary")

asset_summary = pd.DataFrame({
    'Asset': tickers,
    'Annual Return': (annual_returns.values * 100).round(2),
    'Annual Volatility': (annual_vol.values * 100).round(2),
    'Sharpe Ratio': ((annual_returns.values - rf_rate) / annual_vol.values).round(3)
})

col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(asset_summary, use_container_width=True, hide_index=True)
with col2:
    st.metric("Total Assets", len(tickers))
    st.metric("Analysis Period", f"{time_period}")
    st.metric("Risk-Free Rate", f"{rf_rate*100:.2f}%")

# ============================================================================
# SECTION 2: PORTFOLIO OPTIMIZATION
# ============================================================================

st.markdown("---")
st.markdown("## 🚀 Portfolio Optimization")

st.info("Computing optimal portfolios...")

try:
    min_var_weights = optimize_portfolio(annual_returns.values, cov_matrix.values, rf_rate, 'min_var')
    max_sharpe_weights = optimize_portfolio(annual_returns.values, cov_matrix.values, rf_rate, 'sharpe')
    rp_weights = risk_parity_allocation(cov_matrix.values)
    ew_weights = np.array([1.0/len(tickers)]*len(tickers))
    
    if min_var_weights is None or max_sharpe_weights is None:
        st.error("❌ Could not compute optimal portfolios.")
        st.stop()
    
except Exception as e:
    st.error(f"❌ Error in portfolio optimization: {str(e)}")
    st.stop()

def get_portfolio_stats(weights, name):
    ret, vol, sharpe = portfolio_stats(weights, annual_returns.values, cov_matrix.values, rf_rate)
    return {
        'Strategy': name,
        'Return (%)': ret * 100,
        'Volatility (%)': vol * 100,
        'Sharpe Ratio': sharpe,
        'Return/Risk': (ret / vol) if vol > 0 else 0
    }

results = [
    get_portfolio_stats(min_var_weights, 'Minimum Variance'),
    get_portfolio_stats(max_sharpe_weights, 'Maximum Sharpe'),
    get_portfolio_stats(rp_weights, 'Risk Parity'),
    get_portfolio_stats(ew_weights, 'Equal Weight')
]

results_df = pd.DataFrame(results)

st.markdown("### Portfolio Optimization Results")
st.dataframe(results_df.set_index('Strategy'), use_container_width=True)

# ============================================================================
# SECTION 3: ALLOCATION COMPARISON
# ============================================================================

st.markdown("---")
st.markdown("## 📈 Portfolio Allocations")

col1, col2, col3, col4 = st.columns(4)

portfolios = [
    (min_var_weights, 'Minimum Variance', col1),
    (max_sharpe_weights, 'Maximum Sharpe', col2),
    (rp_weights, 'Risk Parity', col3),
    (ew_weights, 'Equal Weight', col4)
]

for weights, name, col in portfolios:
    with col:
        allocation = pd.DataFrame({
            'Asset': tickers,
            'Weight': (weights * 100).round(2)
        })
        allocation = allocation[allocation['Weight'] > 0.1]
        
        fig = go.Figure(data=[go.Pie(
            labels=allocation['Asset'],
            values=allocation['Weight'],
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>',
            marker=dict(
                line=dict(color='#0a0e27', width=2)
            )
        )])
        fig.update_layout(
            title=name,
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', family='Segoe UI')
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ============================================================================
# SECTION 4: EFFICIENT FRONTIER
# ============================================================================

st.markdown("---")
st.markdown("## 📊 Efficient Frontier Analysis")

st.info("Computing efficient frontier...")

try:
    frontier_returns, frontier_vols = compute_efficient_frontier(
        annual_returns.values, cov_matrix.values, rf_rate, n_points=80
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=frontier_vols * 100,
        y=frontier_returns * 100,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='#00d4ff', width=4),
        hovertemplate='Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
    ))
    
    for i, ticker in enumerate(tickers):
        fig.add_trace(go.Scatter(
            x=[annual_vol.values[i] * 100],
            y=[annual_returns.values[i] * 100],
            mode='markers+text',
            name=ticker,
            marker=dict(size=12, opacity=0.8, color='#ffd60a', line=dict(width=2, color='#ffffff')),
            text=[ticker],
            textposition='top center',
            textfont=dict(color='#ffffff', size=10),
            hovertemplate=f'<b>{ticker}</b><br>Vol: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<extra></extra>'
        ))
    
    optimized = [
        (min_var_weights, 'Min Variance', '#10b981'),
        (max_sharpe_weights, 'Max Sharpe', '#ff006e'),
        (rp_weights, 'Risk Parity', '#00d4ff'),
        (ew_weights, 'Equal Weight', '#ffd60a')
    ]
    
    for weights, name, color in optimized:
        ret, vol, sharpe = portfolio_stats(weights, annual_returns.values, cov_matrix.values, rf_rate)
        fig.add_trace(go.Scatter(
            x=[vol * 100],
            y=[ret * 100],
            mode='markers',
            name=name,
            marker=dict(size=20, symbol='star', color=color, line=dict(width=3, color='#ffffff')),
            hovertemplate=f'<b>{name}</b><br>Vol: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<br>Sharpe: {sharpe:.3f}<extra></extra>'
        ))
    
    if len(frontier_vols) > 0:
        max_vol = frontier_vols.max()
        cal_vols = np.linspace(0, max_vol * 1.2, 100)
        sharpe_max = portfolio_stats(max_sharpe_weights, annual_returns.values, cov_matrix.values, rf_rate)[2]
        cal_returns = rf_rate + sharpe_max * cal_vols
        
        fig.add_trace(go.Scatter(
            x=cal_vols * 100,
            y=cal_returns * 100,
            mode='lines',
            name='Capital Allocation Line',
            line=dict(color='rgba(160, 174, 192, 0.5)', width=2, dash='dash'),
            hovertemplate='Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='Efficient Frontier with Optimized Portfolios',
        xaxis_title='Annual Volatility (%)',
        yaxis_title='Annual Return (%)',
        height=600,
        hovermode='closest',
        template='plotly_dark',
        paper_bgcolor='rgba(10, 14, 39, 0.5)',
        plot_bgcolor='rgba(26, 31, 58, 0.3)',
        font=dict(color='#ffffff', family='Segoe UI', size=12),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(42, 48, 85, 0.3)',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='#2a3055'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(42, 48, 85, 0.3)',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='#2a3055'
        ),
        legend=dict(
            bgcolor='rgba(26, 31, 58, 0.8)',
            bordercolor='#2a3055',
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"❌ Error: {str(e)}")

# ============================================================================
# SECTION 5: RISK CONTRIBUTION ANALYSIS
# ============================================================================

st.markdown("---")
st.markdown("## 🎯 Risk Attribution")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Maximum Sharpe Portfolio")
    try:
        port_vol_ms = portfolio_stats(max_sharpe_weights, annual_returns.values, 
                                       cov_matrix.values, rf_rate)[1]
        marginal_contrib_ms = np.dot(cov_matrix.values, max_sharpe_weights)
        rc_ms = max_sharpe_weights * marginal_contrib_ms / (port_vol_ms + 1e-10)
        
        rc_df_ms = pd.DataFrame({
            'Asset': tickers,
            'Weight (%)': (max_sharpe_weights * 100).round(2),
            'Risk Contribution (%)': (rc_ms * 100).round(2)
        })
        
        fig_rc_ms = px.bar(rc_df_ms, x='Asset', y='Risk Contribution (%)',
                           color='Risk Contribution (%)',
                           color_continuous_scale=[[0, '#00d4ff'], [1, '#0099cc']],
                           hover_data=['Weight (%)'])
        fig_rc_ms.update_layout(
            height=400,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', family='Segoe UI')
        )
        st.plotly_chart(fig_rc_ms, use_container_width=True, config={'displayModeBar': False})
    except Exception as e:
        st.error(f"Error: {str(e)}")

with col2:
    st.subheader("Risk Parity Portfolio")
    try:
        port_vol_rp = portfolio_stats(rp_weights, annual_returns.values, 
                                       cov_matrix.values, rf_rate)[1]
        marginal_contrib_rp = np.dot(cov_matrix.values, rp_weights)
        rc_rp = rp_weights * marginal_contrib_rp / (port_vol_rp + 1e-10)
        
        rc_df_rp = pd.DataFrame({
            'Asset': tickers,
            'Weight (%)': (rp_weights * 100).round(2),
            'Risk Contribution (%)': (rc_rp * 100).round(2)
        })
        
        fig_rc_rp = px.bar(rc_df_rp, x='Asset', y='Risk Contribution (%)',
                           color='Risk Contribution (%)',
                           color_continuous_scale=[[0, '#10b981'], [1, '#059669']])
        fig_rc_rp.update_layout(
            height=400,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', family='Segoe UI')
        )
        st.plotly_chart(fig_rc_rp, use_container_width=True, config={'displayModeBar': False})
    except Exception as e:
        st.error(f"Error: {str(e)}")

# ============================================================================
# SECTION 6: COVARIANCE MATRIX HEATMAP
# ============================================================================

st.markdown("---")
st.markdown("## 🔗 Asset Correlations")

try:
    fig_cov = go.Figure(data=go.Heatmap(
        z=cov_matrix.values,
        x=tickers,
        y=tickers,
        colorscale='RdBu',
        zmid=0,
        text=np.round(cov_matrix.values, 3),
        texttemplate='%{text:.2f}',
        textfont={"size": 11, "color": "#ffffff"},
        colorbar=dict(title="Covariance", thickness=20, len=0.7)
    ))
    
    fig_cov.update_layout(
        title='Covariance Matrix (Annualized)',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff', family='Segoe UI'),
        xaxis=dict(showline=True, linewidth=1, linecolor='#2a3055'),
        yaxis=dict(showline=True, linewidth=1, linecolor='#2a3055')
    )
    
    st.plotly_chart(fig_cov, use_container_width=True)
except Exception as e:
    st.error(f"Error: {str(e)}")

# ============================================================================
# SECTION 7: DOWNLOADABLE RESULTS
# ============================================================================

st.markdown("---")
st.markdown("## 💾 Download Results")

try:
    download_data = pd.DataFrame({
        'Strategy': ['Minimum Variance', 'Maximum Sharpe', 'Risk Parity', 'Equal Weight'],
        'Annual Return (%)': results_df['Return (%)'].round(2),
        'Annual Volatility (%)': results_df['Volatility (%)'].round(2),
        'Sharpe Ratio': results_df['Sharpe Ratio'].round(3)
    })
    
    allocations_download = pd.concat([
        pd.DataFrame({'Strategy': 'Minimum Variance', 'Asset': tickers, 'Weight (%)': (min_var_weights*100).round(2)}),
        pd.DataFrame({'Strategy': 'Maximum Sharpe', 'Asset': tickers, 'Weight (%)': (max_sharpe_weights*100).round(2)}),
        pd.DataFrame({'Strategy': 'Risk Parity', 'Asset': tickers, 'Weight (%)': (rp_weights*100).round(2)}),
        pd.DataFrame({'Strategy': 'Equal Weight', 'Asset': tickers, 'Weight (%)': (ew_weights*100).round(2)})
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv1 = download_data.to_csv(index=False)
        st.download_button(
            label="📊 Download Performance Metrics",
            data=csv1,
            file_name="portfolio_performance.csv",
            mime="text/csv"
        )
    
    with col2:
        csv2 = allocations_download.to_csv(index=False)
        st.download_button(
            label="📈 Download Allocations",
            data=csv2,
            file_name="portfolio_allocations.csv",
            mime="text/csv"
        )
except Exception as e:
    st.error(f"Error: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
### 📚 About This Dashboard

**Institutional-grade portfolio optimization** engine implementing:
- Markowitz mean-variance framework
- Efficient frontier generation
- Maximum Sharpe ratio optimization
- Minimum variance allocation
- Risk parity methodology

Used by **BlackRock**, **Vanguard**, **Bridgewater**, **Citadel** and other tier-1 institutions.

---

**Built by:** Thu Nguyen | **Target:** Sydney Finance (IB/PE/Quant)  
**GitHub:** [github.com/thunguyen-debug](https://github.com/thunguyen-debug)  
**Email:** thunguyen5260@gmail.com | **LinkedIn:** [linkedin.com/in/thu-nguyen-00nvtt](https://linkedin.com/in/thu-nguyen-00nvtt)

""")

st.markdown("---")
st.success("✅ Dashboard loaded successfully!")
        font-size: 2.5rem;
        color: #1f4788;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# GENERATE SAMPLE DATA (FALLBACK)
# ============================================================================

@st.cache_data
def generate_sample_data(tickers, days=1825):
    """Generate realistic sample data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Realistic parameters for different asset classes
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
        
        # Geometric Brownian Motion
        returns = np.random.normal(p['mu']/252, p['sigma']/np.sqrt(252), days)
        price = 100 * np.exp(np.cumsum(returns))
        data[ticker] = price
    
    return pd.DataFrame(data, index=dates)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.markdown("# ⚙️ Portfolio Settings")
st.sidebar.markdown("---")

# Data source selection
data_source = st.sidebar.radio(
    "📊 Data Source",
    ["Historical (Yahoo Finance)", "Sample Data (Demo)", "Upload CSV"],
    help="Use historical data from Yahoo Finance, sample data, or upload your own CSV"
)

if data_source == "Historical (Yahoo Finance)" or data_source == "Sample Data (Demo)":
    # Asset selection
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
    
    selected_assets = st.sidebar.multiselect(
        "🏢 Select Assets",
        list(available_assets.keys()),
        default=['US Equities', 'US Bonds', 'International Equities'],
        help="Choose which asset classes to include"
    )
    
    tickers = [available_assets[asset] for asset in selected_assets]
    
    # Time period selection
    time_period = st.sidebar.selectbox(
        "📅 Historical Period",
        ["1 Year", "3 Years", "5 Years", "10 Years"],
        index=2,
        help="How much historical data to use for analysis"
    )
    
    period_map = {
        "1 Year": 1 * 365,
        "3 Years": 3 * 365,
        "5 Years": 5 * 365,
        "10 Years": 10 * 365
    }
    
    days_back = period_map[time_period]
    
else:
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload CSV (returns data)",
        type=['csv'],
        help="CSV with date index and asset columns"
    )
    tickers = None

# Risk-free rate
st.sidebar.markdown("### 📈 Optimization Parameters")
rf_rate = st.sidebar.slider(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=4.5,
    step=0.1,
    help="Current 10-year US Treasury yield"
) / 100

# Covariance estimation method
cov_method = st.sidebar.selectbox(
    "Covariance Estimation",
    ["Sample", "Ledoit-Wolf Shrinkage"],
    help="Ledoit-Wolf is more robust, especially for small samples"
)

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

@st.cache_data
def load_data(tickers, days_back):
    """Download historical price data from Yahoo Finance"""
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
            
        except Exception as e:
            pass
    
    if len(data_list) == 0:
        return None, None
    
    data = pd.concat(data_list, axis=1)
    data.columns = successful_tickers
    data = data.dropna()
    
    return data, successful_tickers

@st.cache_data
def compute_statistics(data):
    """Compute returns, volatility, and covariance"""
    daily_returns = data.pct_change().dropna()
    annual_returns = daily_returns.mean() * 252
    annual_vol = daily_returns.std() * np.sqrt(252)
    cov_matrix = daily_returns.cov() * 252
    
    return daily_returns, annual_returns, annual_vol, cov_matrix

def estimate_covariance_robust(daily_returns):
    """Use Ledoit-Wolf shrinkage for robust covariance estimation"""
    try:
        lw = LedoitWolf()
        cov_shrink, _ = lw.fit(daily_returns)
        return pd.DataFrame(
            cov_shrink * 252,
            index=daily_returns.columns,
            columns=daily_returns.columns
        )
    except:
        return daily_returns.cov() * 252

# ============================================================================
# PORTFOLIO OPTIMIZATION FUNCTIONS
# ============================================================================

def portfolio_stats(weights, returns, cov_matrix, rf_rate):
    """Calculate portfolio statistics"""
    portfolio_return = np.sum(weights * returns)
    portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_vol = np.sqrt(portfolio_var)
    sharpe_ratio = (portfolio_return - rf_rate) / portfolio_vol if portfolio_vol > 0 else 0
    return portfolio_return, portfolio_vol, sharpe_ratio

def negative_sharpe(weights, returns, cov_matrix, rf_rate):
    """Objective function to maximize Sharpe ratio"""
    return -portfolio_stats(weights, returns, cov_matrix, rf_rate)[2]

def portfolio_volatility(weights, returns, cov_matrix, rf_rate):
    """Objective function to minimize volatility"""
    return portfolio_stats(weights, returns, cov_matrix, rf_rate)[1]

def optimize_portfolio(returns, cov_matrix, rf_rate, objective='sharpe'):
    """Run portfolio optimization"""
    n = len(returns)
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    
    if objective == 'sharpe':
        result = minimize(
            negative_sharpe,
            x0=np.array([1/n]*n),
            args=(returns, cov_matrix, rf_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
    elif objective == 'min_var':
        result = minimize(
            portfolio_volatility,
            x0=np.array([1/n]*n),
            args=(returns, cov_matrix, rf_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
    
    return result.x if result.success else None

def risk_parity_allocation(cov_matrix):
    """Compute risk-parity weights (equal risk contribution)"""
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
    """Generate efficient frontier"""
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
        
        result = minimize(
            portfolio_volatility,
            x0=np.array([1/n]*n),
            args=(returns, cov_matrix, rf_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            _, vol, _ = portfolio_stats(result.x, returns, cov_matrix, rf_rate)
            frontier_vols.append(vol)
            frontier_returns.append(target_ret)
    
    return np.array(frontier_returns), np.array(frontier_vols)

# ============================================================================
# MAIN APP
# ============================================================================

# Title
st.markdown('<div class="main-header">🎯 Portfolio Optimizer Dashboard</div>', 
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Institutional-grade mean-variance optimization & efficient frontier analysis</div>',
    unsafe_allow_html=True
)

# Load data
if data_source == "Historical (Yahoo Finance)":
    if not tickers:
        st.warning("⚠️ Please select at least one asset from the sidebar")
        st.stop()
    
    with st.spinner("📥 Loading historical data..."):
        try:
            data, successful_tickers = load_data(tickers, days_back)
            if data is None or len(data) == 0:
                st.warning("⚠️ Could not download from Yahoo Finance. Using sample data instead...")
                data = generate_sample_data(tickers, days_back)
                successful_tickers = tickers
            tickers = successful_tickers
        except Exception as e:
            st.warning("⚠️ Using sample data for demonstration...")
            data = generate_sample_data(tickers, days_back)
            
elif data_source == "Sample Data (Demo)":
    if not tickers:
        st.warning("⚠️ Please select at least one asset from the sidebar")
        st.stop()
    
    st.info("📊 Using realistic sample data for demonstration")
    data = generate_sample_data(tickers, days_back)
    
else:  # Upload CSV
    if uploaded_file is None:
        st.warning("⚠️ Please upload a CSV file")
        st.stop()
    
    try:
        data = pd.read_csv(uploaded_file, index_col=0)
        tickers = list(data.columns)
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {str(e)}")
        st.stop()

# Compute statistics
try:
    daily_returns, annual_returns, annual_vol, cov_matrix = compute_statistics(data)
except Exception as e:
    st.error(f"❌ Error computing statistics: {str(e)}")
    st.stop()

# Apply covariance method
if cov_method == "Ledoit-Wolf Shrinkage":
    try:
        cov_matrix = estimate_covariance_robust(daily_returns)
    except Exception as e:
        st.warning(f"⚠️ Using sample covariance instead")

# ============================================================================
# SECTION 1: ASSET SUMMARY
# ============================================================================

st.markdown("---")
st.markdown("## 📊 Asset Class Summary")

asset_summary = pd.DataFrame({
    'Asset': tickers,
    'Annual Return': (annual_returns.values * 100).round(2),
    'Annual Volatility': (annual_vol.values * 100).round(2),
    'Sharpe Ratio': ((annual_returns.values - rf_rate) / annual_vol.values).round(3)
})

col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(asset_summary, use_container_width=True, hide_index=True)
with col2:
    st.metric("Total Assets", len(tickers))
    st.metric("Analysis Period", f"{time_period}")
    st.metric("Risk-Free Rate", f"{rf_rate*100:.2f}%")

# ============================================================================
# SECTION 2: PORTFOLIO OPTIMIZATION
# ============================================================================

st.markdown("---")
st.markdown("## 🚀 Portfolio Optimization")

st.info("Computing optimal portfolios...")

try:
    min_var_weights = optimize_portfolio(annual_returns.values, cov_matrix.values, rf_rate, 'min_var')
    max_sharpe_weights = optimize_portfolio(annual_returns.values, cov_matrix.values, rf_rate, 'sharpe')
    rp_weights = risk_parity_allocation(cov_matrix.values)
    ew_weights = np.array([1.0/len(tickers)]*len(tickers))
    
    if min_var_weights is None or max_sharpe_weights is None:
        st.error("❌ Could not compute optimal portfolios.")
        st.stop()
    
except Exception as e:
    st.error(f"❌ Error in portfolio optimization: {str(e)}")
    st.stop()

# Compute statistics for each portfolio
def get_portfolio_stats(weights, name):
    ret, vol, sharpe = portfolio_stats(weights, annual_returns.values, cov_matrix.values, rf_rate)
    return {
        'Strategy': name,
        'Return (%)': ret * 100,
        'Volatility (%)': vol * 100,
        'Sharpe Ratio': sharpe,
        'Return/Risk': (ret / vol) if vol > 0 else 0
    }

results = [
    get_portfolio_stats(min_var_weights, 'Minimum Variance'),
    get_portfolio_stats(max_sharpe_weights, 'Maximum Sharpe'),
    get_portfolio_stats(rp_weights, 'Risk Parity'),
    get_portfolio_stats(ew_weights, 'Equal Weight')
]

results_df = pd.DataFrame(results)

# Display results
st.markdown("### Portfolio Optimization Results")
st.dataframe(results_df.set_index('Strategy'), use_container_width=True)

# ============================================================================
# SECTION 3: ALLOCATION COMPARISON
# ============================================================================

st.markdown("---")
st.markdown("## 📈 Portfolio Allocations")

col1, col2, col3, col4 = st.columns(4)

portfolios = [
    (min_var_weights, 'Minimum Variance', col1),
    (max_sharpe_weights, 'Maximum Sharpe', col2),
    (rp_weights, 'Risk Parity', col3),
    (ew_weights, 'Equal Weight', col4)
]

for weights, name, col in portfolios:
    with col:
        allocation = pd.DataFrame({
            'Asset': tickers,
            'Weight': (weights * 100).round(2)
        })
        allocation = allocation[allocation['Weight'] > 0.1]
        
        fig = go.Figure(data=[go.Pie(
            labels=allocation['Asset'],
            values=allocation['Weight'],
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>'
        )])
        fig.update_layout(
            title=name,
            height=400,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SECTION 4: EFFICIENT FRONTIER
# ============================================================================

st.markdown("---")
st.markdown("## 📊 Efficient Frontier Analysis")

st.info("Computing efficient frontier...")

try:
    frontier_returns, frontier_vols = compute_efficient_frontier(
        annual_returns.values, cov_matrix.values, rf_rate, n_points=80
    )
    
    # Create efficient frontier plot
    fig = go.Figure()
    
    # Add frontier
    fig.add_trace(go.Scatter(
        x=frontier_vols * 100,
        y=frontier_returns * 100,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='#1f4788', width=3),
        hovertemplate='Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
    ))
    
    # Add individual assets
    for i, ticker in enumerate(tickers):
        fig.add_trace(go.Scatter(
            x=[annual_vol.values[i] * 100],
            y=[annual_returns.values[i] * 100],
            mode='markers+text',
            name=ticker,
            marker=dict(size=10, opacity=0.7),
            text=[ticker],
            textposition='top center',
            hovertemplate=f'<b>{ticker}</b><br>Vol: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<extra></extra>'
        ))
    
    # Add optimized portfolios
    optimized = [
        (min_var_weights, 'Minimum Variance', '#FF6B6B'),
        (max_sharpe_weights, 'Maximum Sharpe', '#4ECDC4'),
        (rp_weights, 'Risk Parity', '#95E1D3'),
        (ew_weights, 'Equal Weight', '#FFE66D')
    ]
    
    for weights, name, color in optimized:
        ret, vol, sharpe = portfolio_stats(weights, annual_returns.values, cov_matrix.values, rf_rate)
        fig.add_trace(go.Scatter(
            x=[vol * 100],
            y=[ret * 100],
            mode='markers+text',
            name=name,
            marker=dict(size=15, symbol='star', color=color, line=dict(width=2, color='white')),
            text=[name],
            textposition='top center',
            hovertemplate=f'<b>{name}</b><br>Vol: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<br>Sharpe: {sharpe:.3f}<extra></extra>'
        ))
    
    # Add capital allocation line
    if len(frontier_vols) > 0:
        max_vol = frontier_vols.max()
        cal_vols = np.linspace(0, max_vol * 1.2, 100)
        sharpe_max = portfolio_stats(max_sharpe_weights, annual_returns.values, cov_matrix.values, rf_rate)[2]
        cal_returns = rf_rate + sharpe_max * cal_vols
        
        fig.add_trace(go.Scatter(
            x=cal_vols * 100,
            y=cal_returns * 100,
            mode='lines',
            name='Capital Allocation Line',
            line=dict(color='rgba(200, 200, 200, 0.5)', width=2, dash='dash'),
            hovertemplate='Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='Efficient Frontier with Optimized Portfolios',
        xaxis_title='Annual Volatility (%)',
        yaxis_title='Annual Return (%)',
        height=600,
        hovermode='closest',
        template='plotly_white',
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"❌ Error: {str(e)}")

# ============================================================================
# SECTION 5: RISK CONTRIBUTION ANALYSIS
# ============================================================================

st.markdown("---")
st.markdown("## 🎯 Risk Attribution")

col1, col2 = st.columns(2)

# Risk contribution for max sharpe
with col1:
    st.subheader("Maximum Sharpe Portfolio")
    try:
        port_vol_ms = portfolio_stats(max_sharpe_weights, annual_returns.values, 
                                       cov_matrix.values, rf_rate)[1]
        marginal_contrib_ms = np.dot(cov_matrix.values, max_sharpe_weights)
        rc_ms = max_sharpe_weights * marginal_contrib_ms / (port_vol_ms + 1e-10)
        
        rc_df_ms = pd.DataFrame({
            'Asset': tickers,
            'Weight (%)': (max_sharpe_weights * 100).round(2),
            'Risk Contribution (%)': (rc_ms * 100).round(2)
        })
        
        fig_rc_ms = px.bar(rc_df_ms, x='Asset', y='Risk Contribution (%)',
                           title='Risk Contribution by Asset',
                           color='Risk Contribution (%)',
                           color_continuous_scale='Blues')
        fig_rc_ms.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_rc_ms, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Risk contribution for risk parity
with col2:
    st.subheader("Risk Parity Portfolio")
    try:
        port_vol_rp = portfolio_stats(rp_weights, annual_returns.values, 
                                       cov_matrix.values, rf_rate)[1]
        marginal_contrib_rp = np.dot(cov_matrix.values, rp_weights)
        rc_rp = rp_weights * marginal_contrib_rp / (port_vol_rp + 1e-10)
        
        rc_df_rp = pd.DataFrame({
            'Asset': tickers,
            'Weight (%)': (rp_weights * 100).round(2),
            'Risk Contribution (%)': (rc_rp * 100).round(2)
        })
        
        fig_rc_rp = px.bar(rc_df_rp, x='Asset', y='Risk Contribution (%)',
                           title='Risk Contribution by Asset',
                           color='Risk Contribution (%)',
                           color_continuous_scale='Greens')
        fig_rc_rp.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_rc_rp, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")

# ============================================================================
# SECTION 6: COVARIANCE MATRIX HEATMAP
# ============================================================================

st.markdown("---")
st.markdown("## 🔗 Asset Correlations")

try:
    fig_cov = go.Figure(data=go.Heatmap(
        z=cov_matrix.values,
        x=tickers,
        y=tickers,
        colorscale='RdBu',
        zmid=0,
        text=np.round(cov_matrix.values, 3),
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Covariance")
    ))
    
    fig_cov.update_layout(
        title='Covariance Matrix (Annualized)',
        height=500
    )
    
    st.plotly_chart(fig_cov, use_container_width=True)
except Exception as e:
    st.error(f"Error: {str(e)}")

# ============================================================================
# SECTION 7: DOWNLOADABLE RESULTS
# ============================================================================

st.markdown("---")
st.markdown("## 💾 Download Results")

try:
    download_data = pd.DataFrame({
        'Strategy': ['Minimum Variance', 'Maximum Sharpe', 'Risk Parity', 'Equal Weight'],
        'Annual Return (%)': results_df['Return (%)'].round(2),
        'Annual Volatility (%)': results_df['Volatility (%)'].round(2),
        'Sharpe Ratio': results_df['Sharpe Ratio'].round(3)
    })
    
    allocations_download = pd.concat([
        pd.DataFrame({'Strategy': 'Minimum Variance', 'Asset': tickers, 'Weight (%)': (min_var_weights*100).round(2)}),
        pd.DataFrame({'Strategy': 'Maximum Sharpe', 'Asset': tickers, 'Weight (%)': (max_sharpe_weights*100).round(2)}),
        pd.DataFrame({'Strategy': 'Risk Parity', 'Asset': tickers, 'Weight (%)': (rp_weights*100).round(2)}),
        pd.DataFrame({'Strategy': 'Equal Weight', 'Asset': tickers, 'Weight (%)': (ew_weights*100).round(2)})
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv1 = download_data.to_csv(index=False)
        st.download_button(
            label="📊 Download Performance Metrics (CSV)",
            data=csv1,
            file_name="portfolio_performance.csv",
            mime="text/csv"
        )
    
    with col2:
        csv2 = allocations_download.to_csv(index=False)
        st.download_button(
            label="📈 Download Allocations (CSV)",
            data=csv2,
            file_name="portfolio_allocations.csv",
            mime="text/csv"
        )
except Exception as e:
    st.error(f"Error: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
### 📚 About This Dashboard

This dashboard implements **institutional-grade portfolio optimization** used by:
- Asset managers (BlackRock, Vanguard, State Street)
- Hedge funds (Bridgewater, Citadel, Two Sigma)
- Pension funds and endowments

**Key Concepts:**
- **Efficient Frontier**: Set of portfolios that minimize risk for each return level
- **Maximum Sharpe**: Portfolio with best risk-adjusted returns
- **Minimum Variance**: Portfolio with lowest volatility
- **Risk Parity**: Portfolio where each asset contributes equally to risk
- **Capital Allocation Line**: Shows risk-return tradeoff

**Data Source**: Yahoo Finance (or sample data if unavailable)  
**Time Period**: Configurable (1Y, 3Y, 5Y, 10Y)  

---

**Built for:** Finance recruiting (IB, PE, Quant)  
**GitHub**: [github.com/thunguyen-debug/portfolio-optimization-mpt](https://github.com/thunguyen-debug/portfolio-optimization-mpt)  
**Author**: Thu Nguyen | Targeting Sydney finance roles

""")

st.markdown("---")
st.success("✅ Dashboard loaded successfully!")
