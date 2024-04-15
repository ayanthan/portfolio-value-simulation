import numpy as np 
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
import datetime
def get_government_bonds():
    government_bonds = [
        {"face_value": 1, "interest_rate": 0.03, "max_bonds": 1000000000},
        {"face_value": 1, "interest_rate": 0.03, "max_bonds": 500000000000},
        {"face_value": 1400, "interest_rate": 0.02, "max_bonds": 5000},
        # Add more bonds with different interest rates and limits as needed
    ]
    return government_bonds

def calculate_annual_bond_return(remaining_investment):
    bonds_bought = 0
    annual_bond_return = 0  # Initialize annual bond return

    government_bonds = get_government_bonds()
    sorted_government_bonds = sorted(government_bonds, key=lambda x: x["interest_rate"], reverse=True)
    
    for _, bond in enumerate(sorted_government_bonds):
        bond_price = bond['face_value']
        max_bonds = bond['max_bonds']
        
        while remaining_investment >= bond_price and bonds_bought < max_bonds:
            bonds_bought += 1
            remaining_investment -= bond_price
            # Calculate annual bond returns
            annual_bond_return += bond["interest_rate"] * bond_price  # Accumulate the annual bond return
        
    return annual_bond_return

def adj_close_price_df():
    #time range
    years = 20
    endDate = dt.datetime.now()
    startDate = endDate - dt.timedelta(days = 365*years)

    #create a list of tickers
    tickers = ['SPY', 'GLD', 'QQQ', 'VII']
 
    #Download 
    adj_close_df = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(ticker, start = startDate, end = endDate)
        adj_close_df[ticker] = data['Adj Close']
    
    #adj_close_df['bond_return'] = calculate_annual_bond_return(bond_investment)/252
    adj_close_df = adj_close_df.dropna()
    return adj_close_df

def log_return(adj_close_df):
    #daily log_return and drop Na #using log bc log returns are additives
    log_returns = np.log(adj_close_df/adj_close_df.shift(1))
    log_returns = log_returns.dropna()
    return log_returns

def is_pos_def(cov_matrix):
    return np.all(np.linalg.eigvals(cov_matrix) >= 0)
    
#create a function that will be used to calculate portfolio expected return
def expected_return(weights, log_returns):
    return np.sum(log_returns.mean()*weights)

#create a function that will be used to calculate portfolio standard deviation#create a function that will be used to calculate portfolio expected return
def standard_deviation(weights, cov_matrix):
    print(cov_matrix)
    if is_pos_def(cov_matrix):
        variance = weights.T @ cov_matrix @ weights
        std_deviation = np.sqrt(variance)
        print(f"Portfolio variance: {variance:.3}")
        return std_deviation
    else:
        print("Error: Covariance matrix is not positive semidefinite.")
        return None

#create a function that gives us random Z-score based on normal distribution
def random_z_score():
    return np.random.normal(0,1)

def scenario_gain_loss(portfolio_value,portfolio_expected_return,portfolio_std_dev, z_score, days,confidence_interval):
    return portfolio_value * portfolio_expected_return * days + portfolio_value * portfolio_std_dev * z_score * np.sqrt(days)
#portfolio_value * portfolio_expected_return*days: tells us what would be ur typicall expected return over this amount of days {days}
#portfolio_value * portfolio_std_dev * z_score * np.sqrt(days): adding the volatile element


def Value_at_Risk(simulations,portfolio_value,portfolio_expected_return,portfolio_std_dev, z_score, days,confidence_interval):
    scenarioReturn = [] 
    
    for i in range(simulations):
        z_score = random_z_score()
        scenarioReturn.append(scenario_gain_loss(portfolio_value,portfolio_expected_return,portfolio_std_dev, z_score, days,confidence_interval))
        
    VaR = -np.percentile(scenarioReturn, 100*(1-confidence_interval))
    print(VaR)
    return VaR,scenarioReturn

def plot_histogram_VaR(VaR,scenarioReturn,portfolio_years, confidence_interval):
    #plot the result
    plt.hist(scenarioReturn, bins=50, density=True)
    plt.xlabel('Scenario Gain/Loss ($)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Portfolio Gain/Loss Over {portfolio_years} years')
    plt.axvline(-VaR, color = 'r', linestyle='dashed', linewidth =2, label=f'VaR at {confidence_interval:.0%} confidence interval')
    plt.legend()
    plt.show()

def main():
    portfolio_value = 10000
    weights = np.array([0.7,0.1,0.1,0.1])
    adj_close_df = adj_close_price_df()
    log_returns = log_return(adj_close_df)
    z_score = random_z_score()
    
    #create annualized covariance matrix for all the securities
    cov_matrix = log_returns.cov()

    portfolio_expected_return = expected_return(weights, log_returns)
    portfolio_std_dev = standard_deviation(weights, cov_matrix)

    #create a function to calculate scenarioGainloss
    portfolio_years = 30
    days = portfolio_years #assuming 252 trading days
    confidence_interval = 0.95 #higher the confidence interval is the greater the value at risk will bc move further out the tail of the distribution
    simulations = 10000
    
    VaR,scenarioReturn = Value_at_Risk(simulations,portfolio_value,portfolio_expected_return,portfolio_std_dev,z_score,days,confidence_interval)
    plot_histogram_VaR(VaR,scenarioReturn,portfolio_years, confidence_interval)
if __name__ == '__main__':
    main()


