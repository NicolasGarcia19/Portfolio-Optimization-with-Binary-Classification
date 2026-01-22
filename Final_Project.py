import yfinance as yf
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


'''
Download Data
'''
print(yf.__version__)

tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN', 'WMT', 'PG', 'JPM', 'BAC', 'JNJ', 'UNH', 'XOM', 'CVX', 'NEE']

data = yf.download(tickers, start = '2018-01-01', end = '2024-12-01')
prices = data['Close']
returns = prices.pct_change()
volume = data['Volume']

print("Missing data per ticker:")
print(prices.isnull().sum())

problematic = prices.columns[prices.isnull().sum() / len(prices) > 0.05]
if len(problematic) > 0:
    print(f"Warning: these tickers have >5% missing data: {list(problematic)}")

vix = yf.download('^VIX', start = '2018-01-01', end = '2024-12-01')['Close']
sp500 = yf.download('^GSPC', start = '2018-01-01', end = '2024-12-01')['Close']

'''
calculate features
'''

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window = period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window = period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_all_features(prices, returns, volume, sp500, vix):
    #Returns a dataframe with one row per (date,ticker) pair

    features = pd.DataFrame(index = prices.index)

    #price based metrics

    #5 day return
    features['ret_5d'] = returns.rolling(5).sum()
    #10 day return
    features['ret_10d'] = returns.rolling(10).sum()
    #20 day momentum
    features['momentum_20d'] = returns.rolling(20).sum()
    #volatility
    features['volatility_20d'] = returns.rolling(20).std()
    #price relative to 20 day moving average
    ma_20 = prices.rolling(20).mean()
    features['price_vs_ma20'] = (prices - ma_20) / ma_20

    #technical indicators

    #RSI
    features['rsi'] = compute_rsi(prices, period = 14)

    #volume based metrics

    #volume ratio
    volume_ma_20 = volume.rolling(20).mean()
    features['volume_ratio'] = volume / volume_ma_20
    #short term volume trend
    features['volume_trend'] = volume.pct_change(5)

    #market context

    #S&P 500 recent 5 day return
    features['sp500_ret'] = sp500.pct_change().rolling(5).sum()
    #VIX level
    features['vix_level'] = vix


    return features

all_features = {}
for ticker in tickers:
    features_df = compute_all_features(prices[ticker], returns[ticker], volume[ticker], sp500, vix)
    all_features[ticker] = features_df

print(f"Features computed for {len(tickers)} stocks")
print(f"Feature Names: {list(features_df.columns)}")

next_return = returns.shift(-1)
labels = pd.DataFrame(np.where(next_return > 0, 1, -1), index=returns.index, columns=returns.columns)

'''
Prepare data for optimization
'''

def prep_data(all_features, labels, tickers):
    X_list = []
    Y_list = []
    date_list = []
    stock_list = []

    for ticker in tickers:
        feat = all_features[ticker].copy()
        lab = labels[ticker]

        valid_idx = feat.dropna().index

        X_list.append(feat.loc[valid_idx])
        Y_list.append(lab[valid_idx])
        date_list.extend(valid_idx)
        stock_list.extend([ticker] * len(valid_idx))
    
    X = pd.concat(X_list, axis = 0)
    Y = pd.concat(Y_list)

    return X, Y, date_list, stock_list

X, Y, dates, stock_names = prep_data(all_features, labels, tickers)

print(f"\nTotal samples (date x ticker pairs): {len(X)}")

'''
split data into training and testing
'''

split_date = '2022-12-31'

date_index = pd.DatetimeIndex(dates)

train_mask = date_index <= split_date
test_mask = date_index > split_date

X_train = X[train_mask].values
Y_train = Y[train_mask].values
X_test = X[test_mask].values
Y_test = Y[test_mask].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


'''
Binary classifier: Stage 1
'''

print("\n=== Training Binary Classifier ===")

n_samples, n_features = X_train_scaled.shape

n_pos = (Y_train == 1).sum()
n_neg = (Y_train == -1).sum()
weight_pos = n_samples / (2 * n_pos)
weight_neg = n_samples / (2 * n_neg)

print(f"Class weights: +1: {weight_pos:.3f}, -1: {weight_neg:.3f}")

sample_weights = np.where(Y_train  == 1, weight_pos, weight_neg)


#Decision Variables
w = cp.Variable(n_features)
b = cp.Variable()

#setup hinge loss function
margins = cp.multiply(Y_train, X_train_scaled @ w + b)
hinge_loss = cp.sum(cp.pos(1 - margins))
logistic_loss = cp.sum(cp.logistic(-margins))
weighted_logistic_loss = cp.sum(cp.multiply(sample_weights, cp.logistic(-margins)))



#L2 regularization, helps prevent overfitting
lambda_reg = 6e-5
regularization = lambda_reg * cp.sum_squares(w) + cp.square(b)

#Complete Objective
objective = cp.Minimize(weighted_logistic_loss + regularization)


#Solve problem
problem = cp.Problem(objective)
problem.solve(solver = cp.SCS, verbose = False)

print(f"Solver status: {problem.status}")
print(f"Optimal Value: {problem.value:.2f}")

#Get the trained parameters
w_trained = w.value
b_trained = b.value

'''
Evalaute Success of the Classifier
'''

def eval_classifier(X, Y, w, b):
    margins = X @ w + b
    Y_pred = np.sign(margins)

    #handle 0 margins just in case
    Y_pred[Y_pred == 0] = 1

    accuracy = (Y_pred == Y).mean()

    true_pos = ((Y_pred == 1) & (Y == 1)).sum()
    pred_pos = (Y_pred == 1).sum()
    actual_pos = (Y == 1).sum()

    precision = true_pos / pred_pos if pred_pos > 0 else 0
    recall = true_pos / actual_pos if actual_pos > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'predictions': Y_pred,
        'margins': margins
    } 


train_results = eval_classifier(X_train_scaled, Y_train, w_trained, b_trained)
test_results = eval_classifier(X_test_scaled, Y_test, w_trained, b_trained)

print(f"\n=== Classifier Performance ===")
print(f"Training Accuracy: {train_results['accuracy']:.3f}")
print(f"Testing Accuracy: {test_results['accuracy']:.3f}")
print(f"Test Precision: {test_results['precision']:.3f}")
print(f"Test Recall: {test_results['recall']:.3f}")


'''
Portfolio Optimization: Stage 2
'''

print("\n==== Portfolio Optimization ===")

test_dates_df = pd.DataFrame({'date': dates, 'ticker': stock_names})
test_dates_df = test_dates_df[test_mask].copy()
test_dates_unique = sorted(test_dates_df['date'].unique())

print(f"Backtesting on {len(test_dates_unique)} trading days")

#storage
portfolio_results = {
    'dates': [],
    'classifier_returns': [],
    'equal_weight_returns': [],
    'return_only_returns': []
}

lambda_penalty = 0.1

for i, date in enumerate(test_dates_unique[20:-1]):
    if i % 50 == 0:
        print(f"Processing day {i}/{len(test_dates_unique)-21}...")
    
    features_list = []
    valid_tickers = []

    for ticker in tickers:
        feat = all_features[ticker]
        if date in feat.index:
            features_list.append(feat.loc[date].values)
            valid_tickers.append(ticker)
    if len(features_list) < 5:
        continue

    n_assets = len(valid_tickers)

    X_date = np.array(features_list)
    X_date_scaled = scaler.transform(X_date)

    margins = X_date_scaled @ w_trained + b_trained
    risk_scores = 1 / (1 + np.exp(-margins))

    #estimating the expected returns
    expected_returns = np.array([
        all_features[ticker].loc[date]['momentum_20d'] for ticker in valid_tickers
    ])

    w = cp.Variable(n_assets)

    return_term = expected_returns @ w
    risk_term = lambda_penalty * (1-risk_scores) @ w

    objective = cp.Maximize(return_term - risk_term)

    constraints = [cp.sum(w) == 1, w >= 0]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver = cp.SCS, verbose = False)
    if problem.status != 'optimal':
        continue
    classifier_weights = w.value

    #set baselines, these baselines will help me see if my optimization pipeline actually helps
    equal_weights = np.ones(n_assets) / n_assets

    w_baseline = cp.Variable(n_assets)
    obj_baseline = cp.Maximize(expected_returns @ w_baseline)
    constraints_baseline = [cp.sum(w_baseline) == 1, w_baseline >= 0]
    prob_baseline = cp.Problem(obj_baseline, constraints_baseline)
    prob_baseline.solve(solver = cp.SCS, verbose = False)

    if prob_baseline.status == 'optimal':
        return_only_weights = w_baseline.value
    else:
        return_only_weights = equal_weights

    #calculate next day returns
    next_date_idx = test_dates_unique.index(date) + 1
    if next_date_idx >= len(test_dates_unique):
        continue
    next_date = test_dates_unique[next_date_idx]

    actual_returns = np.array([
        returns[ticker].loc[next_date] if next_date in returns[ticker].index else 0 for ticker in valid_tickers
    ])

    classifier_ret = classifier_weights @ actual_returns
    equal_ret = equal_weights @ actual_returns
    return_only_ret = return_only_weights @ actual_returns

    #store results
    portfolio_results['dates'].append(date)
    portfolio_results['classifier_returns'].append(classifier_ret)
    portfolio_results['equal_weight_returns'].append(equal_ret)
    portfolio_results['return_only_returns'].append(return_only_ret)

print(f"\n Backtesting Complete: {len(portfolio_results['dates'])} days")

#analyze results 
results_df = pd.DataFrame(portfolio_results)
results_df['dates'] = pd.to_datetime(results_df['dates'])
results_df = results_df.set_index('dates')

results_df['classifier_cumulative'] = (1 + results_df['classifier_returns']).cumprod()
results_df['equal_weight_cumulative'] = (1 + results_df['equal_weight_returns']).cumprod()
results_df['return_only_cumulative'] = (1 + results_df['return_only_returns']).cumprod()

def calculate_metrics(returns):
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        'Total Return': f"{total_return*100:.2f}%",
        'Annual Return': f"{annual_return*100:.2f}%",
        'Sharpe Ratio': f"{sharpe:.3f}",
        'Max Drawdown': f"{max_drawdown*100:.2f}%",
        'Daily Volatility': f"{returns.std()*100:.3f}%"
    }

print("\nClassifier-Based (Risk-Penalized):")
for k, v in calculate_metrics(results_df['classifier_returns']).items():
    print(f"  {k}: {v}")

print("\nEqual-Weight Baseline:")
for k, v in calculate_metrics(results_df['equal_weight_returns']).items():
    print(f"  {k}: {v}")

print("\nReturn-Only Baseline (No Classifier):")
for k, v in calculate_metrics(results_df['return_only_returns']).items():
    print(f"  {k}: {v}")


'''
Visualization
'''

plt.figure(figsize=(12, 6))
plt.plot(results_df.index, results_df['classifier_cumulative'], 
         label='Classifier-Based (Risk-Penalized)', linewidth=2.5, color='blue')
plt.plot(results_df.index, results_df['equal_weight_cumulative'], 
         label='Equal-Weight', linewidth=2, linestyle='--', color='green')
plt.plot(results_df.index, results_df['return_only_cumulative'], 
         label='Return-Only (No Classifier)', linewidth=2, linestyle='--', color='red')

plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Return', fontsize=12)
plt.title('Portfolio Performance Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ECE 4800/Final Project/portfolio_performance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n Results saved to 'portfolio_performance.png'")
print("\n=== Pipeline Complete! ===")

