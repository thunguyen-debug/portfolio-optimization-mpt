# Portfolio Optimizer Dashboard

Institutional-grade mean-variance optimization with interactive web interface inspired by modern fintech design.

**Live Demo:** [Streamlit App](https://portfolio-optimizer.streamlit.app)

## Features

- **4 Optimization Strategies**
  - Minimum Variance (lowest risk portfolio)
  - Maximum Sharpe (best risk-adjusted returns)
  - Risk Parity (equal risk contribution)
  - Equal Weight (baseline comparison)

- **Efficient Frontier** - Interactive Plotly visualization with:
  - Efficient frontier curve
  - Individual asset positions
  - Optimized portfolio markers
  - Capital Allocation Line

- **Risk Attribution** - Contribution analysis showing:
  - Asset-level risk contribution
  - Marginal risk impact
  - Diversification effects

- **Asset Correlations** - Heatmap visualization of:
  - Pairwise correlations
  - Covariance structure
  - Diversification opportunities

- **Flexible Data Sources**
  - Yahoo Finance (real historical data)
  - Sample data (demo mode)
  - CSV upload (custom data)

- **Download Results** - Export:
  - Performance metrics (returns, volatility, Sharpe)
  - Portfolio allocations by strategy

## Tech Stack

- **Framework:** Streamlit (web UI)
- **Numerical Computing:** NumPy, SciPy, Pandas
- **Optimization:** SciPy minimize (SLSQP solver)
- **Visualization:** Plotly (interactive charts)
- **Covariance Estimation:** Scikit-learn (Ledoit-Wolf shrinkage)
- **Data:** yfinance (historical prices)

## Installation

```bash
git clone https://github.com/thunguyen-debug/portfolio-optimization-mpt.git
cd portfolio-optimization-mpt
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501

## Usage

1. **Select Data Source**
   - Historical: Live Yahoo Finance data (8 asset classes)
   - Sample: Realistic synthetic data (demo mode)
   - Upload CSV: Your own price/returns data

2. **Choose Assets** - Select from:
   - US Equities (SPY)
   - International Equities (EFA)
   - Emerging Markets (EEM)
   - US Bonds (BND)
   - International Bonds (BNDX)
   - Real Estate REITs (VNQ)
   - Commodities (GSG)
   - Gold (GLD)

3. **Set Parameters**
   - Historical period (1, 3, 5, or 10 years)
   - Risk-free rate (default 4.5%)
   - Covariance method (Sample or Ledoit-Wolf)

4. **View Results**
   - Asset summary table
   - Optimization results (4 strategies)
   - Portfolio allocations (pie charts)
   - Efficient frontier (interactive plot)
   - Risk attribution (bar charts)
   - Correlation heatmap

5. **Download**
   - Performance metrics CSV
   - Portfolio allocations CSV

## Methodology

### Markowitz Mean-Variance Framework

The dashboard implements the classical Markowitz portfolio optimization framework:

**Inputs:**
- Historical asset returns (daily)
- Covariance matrix (annualized)

**Optimization:**
- Minimize portfolio volatility subject to return target
- Maximize Sharpe ratio (risk-adjusted returns)
- Equal risk contribution (risk parity)

**Constraints:**
- Weights sum to 1.0 (fully invested)
- No shorting (weights ≥ 0)

### Key Calculations

**Portfolio Return:**
R_p = Σ(w_i * r_i)
**Portfolio Volatility:**
σ_p = sqrt(w^T * Σ * w)
**Sharpe Ratio:**
Sharpe = (R_p - R_f) / σ_p
Where:
- w = weight vector
- r = expected returns
- Σ = covariance matrix
- R_f = risk-free rate

### Robust Covariance Estimation

Optional Ledoit-Wolf shrinkage reduces estimation error, especially important for:
- Small sample sizes
- High-dimensional problems
- Stressed market conditions

## Backtesting Results

**Sample: 5-Year Historical (SPY, BND, EFA)**

| Strategy | Return | Volatility | Sharpe |
|----------|--------|-----------|--------|
| Min Variance | 4.81% | 4.59% | 0.067 |
| Max Sharpe | 21.35% | 14.87% | 1.133 |
| Risk Parity | 8.48% | 11.60% | 0.343 |
| Equal Weight | 7.19% | 7.85% | 0.342 |

## Limitations & Future Work

**Current Limitations:**
- No transaction costs (bid-ask spreads, commissions)
- Historical returns may not predict future performance
- Covariance estimated from historical data (unstable in stress)
- Long-only constraints (no hedging)
- Deterministic (no scenario analysis)

**Future Enhancements:**
- Transaction cost modeling
- Multi-period optimization (dynamic)
- Constraints on sector/factor exposure
- Short-sale capability
- Monte Carlo scenario analysis
- Robust optimization (uncertainty sets)

## Requirements
streamlit==1.35.0
pandas==2.1.0
numpy==1.24.3
scipy==1.11.2
plotly==5.17.0
yfinance==0.2.32
scikit-learn==1.3.0

## File Structure
portfolio-optimization-mpt/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── .streamlit/
└── config.toml                 # Streamlit theme config
## Performance Notes

- Data download: ~2-5 seconds (Yahoo Finance)
- Covariance computation: <1 second
- Optimization (3 strategies): 1-3 seconds
- Efficient frontier (80 points): 3-5 seconds
- Total load time: ~10 seconds (cold start)

## References

**Core Theory:**
- Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance*
- Sharpe, W. (1964). "Capital Asset Prices: A Theory of Market Equilibrium"

**Implementation:**
- Ledoit, O. & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix"
- Boyd, S., Parikh, N., Chu, E., et al. (2011). "Distributed Optimization and Statistical Learning"

**Practical Considerations:**
- Meucci, A. (2005). "Risk and Asset Allocation" (Springer)
- Clarke, R., de Silva, H., Thorley, S. (2016). "Fundamentals of Efficient Factor Investing"

## Author

**Thu Nguyen**
- Email: thunguyen5260@gmail.com
- GitHub: [thunguyen-debug](https://github.com/thunguyen-debug)
- LinkedIn: [linkedin.com/in/thu-nguyen-00nvtt](https://linkedin.com/in/thu-nguyen-00nvtt)
- Location: Sydney, Australia

## License

MIT License - Use freely for educational and commercial purposes.

## Disclaimer

This tool is for educational purposes. Past performance does not guarantee future results. Portfolio optimization is one component of a comprehensive investment strategy. Consult with a financial advisor before making investment decisions.
