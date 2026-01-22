Two-Stage ML Portfolio Optimization

This project implements a two-stage quantitative investing pipeline combining supervised learning and convex optimization.

Summary
  Stage 1: Train a regularized logistic classifier to predict next-day stock direction using technical, volume, and market features.
  Stage 2: Use classifier confidence as a risk signal in a daily long-only portfolio optimization.
The strategy is backtested on U.S. large-cap equities from 2018 to 2024 and compared against:
  Equal-weight portfolio
  Return-only optimization baseline


Data
  Equity prices and volume from yfinance
  Market indicators: S&P 500 (^GSPC) and VIX (^VIX)
  14 large-cap U.S. stocks across multiple sectors
  
Features
  Short- and medium-term returns and momentum
  Rolling volatility
  RSI
  Volume trends
  Market context (SP500 returns, VIX)
  
Evaluation
  Performance is measured using cumulative returns, Sharpe ratio, and drawdown. Results are visualized and saved as portfolio_performance.png.
  
Dependencies
  yfinance, numpy, pandas, cvxpy, scikit-learn, matplotlib

Notes
  This is a research and educational project. Transaction costs and slippage are not modeled.
