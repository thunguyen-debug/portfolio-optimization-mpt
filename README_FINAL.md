# 🎯 Portfolio Optimizer Dashboard

**Institutional-grade mean-variance optimization & efficient frontier analysis**

Interactive Streamlit dashboard for portfolio construction, backtesting, and stress testing.

---

## 🚀 Try It Live

**[OPEN INTERACTIVE DASHBOARD →](https://portfolio-optimizer-app.streamlit.app)**

Play with:
- 8 asset classes (SPY, EFA, EEM, BND, BNDX, VNQ, GSG, GLD)
- 4 optimization strategies (Min Var, Max Sharpe, Risk Parity, Equal Weight)
- 1-10 year historical periods
- Real-time efficient frontier visualization

---

## ✨ Features

### 📊 Portfolio Optimization
- **Mean-Variance Optimization** (Markowitz, 1952)
- **Efficient Frontier** - 100 optimal portfolios
- **Maximum Sharpe Ratio** - Best risk-adjusted returns
- **Minimum Variance** - Lowest risk allocation
- **Risk Parity** - Equal risk contribution per asset
- **Equal Weight** - Simple 1/n benchmark

### 📈 Analysis & Visualization
- Real-time efficient frontier plots
- Portfolio allocation pie charts
- Risk attribution analysis
- Asset correlation heatmap
- Covariance matrix visualization
- Interactive Plotly charts

### 💾 Data & Computation
- Yahoo Finance integration (8 asset classes)
- CSV upload support
- 1-10 year historical periods
- Ledoit-Wolf shrinkage covariance estimation
- SLSQP optimization solver

### 📥 Results
- Download performance metrics (CSV)
- Download allocations (CSV)
- Compare strategies side-by-side

---

## 🎓 Educational Value

Learn institutional portfolio construction:
- How asset managers optimize capital allocation
- Mean-variance framework (Markowitz)
- Efficient frontier concept
- Sharpe ratio & risk-adjusted returns
- Risk parity allocation method
- Covariance matrix & correlation analysis

**This is the exact framework used by:**
- BlackRock, Vanguard, State Street (Asset Management)
- Citadel, Two Sigma, Bridgewater (Hedge Funds)
- JP Morgan, Goldman Sachs (Prop Trading)
- Pension funds, endowments (Institutional Investors)

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

```bash
# Clone repository
git clone https://github.com/thunguyen-debug/portfolio-optimizer-app.git
cd portfolio-optimizer-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py

# Visit http://localhost:8501
```

### Usage
1. Select assets from sidebar
2. Choose time period (1Y, 3Y, 5Y, 10Y)
3. Adjust risk-free rate if needed
4. Select covariance method (Sample or Ledoit-Wolf)
5. View optimization results
6. Explore efficient frontier
7. Download results as CSV

---

## 📊 Dashboard Sections

### 1. Asset Class Summary
Table showing:
- Annual return
- Annual volatility
- Sharpe ratio
For each asset class (last 1-10 years)

### 2. Portfolio Optimization Results
Comparison table:
- Minimum Variance strategy
- Maximum Sharpe strategy
- Risk Parity strategy
- Equal Weight benchmark

With metrics: Return, Volatility, Sharpe Ratio, Return/Risk

### 3. Portfolio Allocations
Four pie charts showing:
- Weight distribution by asset
- Color-coded by strategy
- Interactive hover details

### 4. Efficient Frontier
Main visualization:
- 100 optimal portfolios on frontier
- Individual asset positions
- Optimized portfolios (highlighted with stars)
- Capital allocation line (tangent to frontier)
- Interactive tooltips

### 5. Risk Attribution
Two bar charts:
- Maximum Sharpe: Risk contribution by asset
- Risk Parity: Risk contribution by asset
Shows how each asset drives portfolio risk

### 6. Correlation Heatmap
Covariance matrix visualization:
- Asset pair correlations
- Color scale (blue = positive, red = negative)
- Numerical values in cells

### 7. Download Results
CSV exports:
- Performance metrics (return, vol, Sharpe for each strategy)
- Allocations (weights for each asset, for each strategy)

---

## 🧮 Mathematical Framework

### Portfolio Return
```
μ_p = Σ w_i × μ_i
```
Expected return is weighted sum of asset returns.

### Portfolio Volatility
```
σ_p = √(w^T × Σ × w)
```
Portfolio risk via variance-covariance matrix. Includes diversification benefit.

### Sharpe Ratio
```
SR = (μ_p - r_f) / σ_p
```
Excess return per unit of risk. Higher = better risk-adjusted performance.

### Efficient Frontier
Set of portfolios that minimize variance for each target return level.
Solves: `min σ_p² subject to μ_p = target, Σw_i = 1`

### Risk Contribution
```
RC_i = w_i × (Σ*w)_i / σ_p
```
How much each asset contributes to portfolio risk.

### Risk Parity
Allocate so each asset contributes equally to portfolio risk:
```
RC_i = σ_p / n for all i
```

---

## 💡 For Recruiters & Interviewers

### Why This Project Matters

**This demonstrates:**
✅ Understanding of institutional finance  
✅ Quantitative analysis & optimization  
✅ Software engineering (clean, professional code)  
✅ Financial domain knowledge  
✅ User experience design  
✅ Data visualization  

### What This Solves

**Investment Banking**: Understand client capital allocation constraints  
**Private Equity**: Portfolio stress testing & capital structure optimization  
**Quantitative Finance**: Core infrastructure for portfolio construction  
**Asset Management**: How factor/smart-beta portfolios are built  

### Interview Talking Points

> "I built an institutional-grade portfolio optimization engine implementing 
> Markowitz mean-variance optimization and efficient frontier generation. 
> The dashboard allows real-time portfolio construction with 4 optimization 
> strategies, risk attribution analysis, and stress testing. This is the exact 
> framework used by BlackRock, Vanguard, and Bridgewater. 
> 
> The technical implementation includes SLSQP solver for constrained optimization, 
> Ledoit-Wolf shrinkage for covariance estimation, and Plotly for interactive 
> visualization. I've identified institutional gaps (no transaction costs, 
> unstable covariance in crises) and know how to fix them with robust methods."

---

## 🏗️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | Streamlit 1.28+ | Interactive web app |
| **Numerical** | NumPy, SciPy | Math & optimization |
| **Data** | Pandas | Data manipulation |
| **Optimization** | SciPy SLSQP | Portfolio optimization |
| **Covariance** | Scikit-learn | Ledoit-Wolf shrinkage |
| **Market Data** | YFinance | Yahoo Finance API |
| **Visualization** | Plotly | Interactive charts |
| **Deployment** | Streamlit Cloud | Free hosting |

---

## 🔧 Advanced Features

### Covariance Estimation Methods
- **Sample**: Direct empirical covariance (standard)
- **Ledoit-Wolf**: Shrinkage estimator (more robust, reduces noise)

Ledoit-Wolf is preferred when:
- Small sample sizes
- High-dimensional covariance
- Tail risk analysis
- Institutional production systems

### Optimization Solver
**SLSQP** (Sequential Least Squares Programming):
- Handles non-linear constraints (Σw_i = 1)
- Handles bounds (0 ≤ w_i ≤ 1)
- Finds locally optimal solutions
- Good for portfolio construction

---

## 📈 Known Limitations & Future Work

### Current Limitations
- ⚠️ No transaction costs modeled (real drag: 30-100 bps annually)
- ⚠️ Fixed weights (no rebalancing drift in backtest)
- ⚠️ Historical covariance (unstable in crises)
- ⚠️ Risk-Free Rate fixed at 4.5%
- ⚠️ No position constraints (max 15% per asset)

### Production Upgrades
- [ ] Transaction-cost modeling
- [ ] Dynamic rebalancing simulation
- [ ] Rolling covariance windows
- [ ] Time-varying risk-free rate
- [ ] Position size constraints
- [ ] Black-Litterman model
- [ ] Monte Carlo simulation
- [ ] Machine learning factor models

---

## 📚 References

**Markowitz, H. (1952)** - Portfolio Selection  
*The Journal of Finance*

**Sharpe, W. (1966)** - Mutual Fund Performance  
*Journal of Business*

**Bridgewater Associates (2005)** - The All Weather Story  
*Risk Parity approach*

**Ledoit, O. & Wolf, M. (2004)** - Honey, I Shrunk the Sample Covariance Matrix  
*Shrinkage estimation methods*

---

## 👤 Author

**Thu Nguyen**

📍 **Location**: Ho Chi Minh City, Vietnam → Sydney, Australia  
📅 **Target Start**: February 2026 (University of Sydney)  

🎯 **Recruiting Focus**:
- Investment Banking (Capital Markets, M&A, Debt Structuring)
- Private Equity (Portfolio Management, Capital Allocation)
- Quantitative Finance (Equity Research, Trading, Risk)

**Experience**:
- Portfolio optimization & institutional risk management
- Financial modeling & quantitative analysis
- Python development & data visualization
- Content creation (Foglit Piano, Science Inside You)
- Entrepreneurial finance (Coffee shop business analysis)

**Interests**:
- Institutional portfolio construction
- Quantitative finance & algorithmic trading
- Financial technology
- Alternative investments

**Contact**:
- 📧 **Email**: thunguyen5260@gmail.com
- 🔗 **LinkedIn**: [linkedin.com/in/thu-nguyen-00nvtt](https://www.linkedin.com/in/thu-nguyen-00nvtt)
- 🐙 **GitHub**: [github.com/thunguyen-debug](https://github.com/thunguyen-debug)

---

## 🤝 Connect

Interested in discussing:
- Portfolio construction & risk management
- Quantitative finance career paths
- IB/PE/Quant recruiting
- Financial modeling
- Sydney finance market

**Reach out!** Always happy to discuss finance, optimization, and technology.

---

## 📄 License

MIT License - Free to use, modify, and distribute

---

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - App framework
- [Plotly](https://plotly.com/) - Visualization
- [YFinance](https://finance.yahoo.com/) - Market data
- [SciPy](https://scipy.org/) - Optimization
- [Pandas](https://pandas.pydata.org/) - Data analysis

---

**Last Updated**: June 30, 2026

**Version**: 1.0 (Production Ready)

---

🚀 **[Try the Dashboard](https://portfolio-optimizer-app.streamlit.app)**
