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

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
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
