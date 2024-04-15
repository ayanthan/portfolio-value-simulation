from account import Account
from portfolio import *
from value_at_risk import *
from company import Company
from models_function import *
from plot_functions import *
import numpy as np
import csv
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

def calculate_each_stock_return(current_stock_price, previous_stock_price):
    return ((current_stock_price - previous_stock_price) / previous_stock_price)

def calculate_total_stock_return(year, stock_investment_all_companies, stock_returns_all_companies, portfolio_time):
    # Check if year is within valid range
    if year < 0 or year >= portfolio_time:
        raise ValueError("Invalid year value")

    # Calculate total stock component returns for all companies
    total_stock_return = 0
    for i in range(len(stock_investment_all_companies)):
        # Check if the current year is within the available stock return data
        number_of_stock_returns_data = len(stock_returns_all_companies[i])
        if year < number_of_stock_returns_data:
            total_stock_return += stock_investment_all_companies[i] * stock_returns_all_companies[i][year]

    return total_stock_return

    
def get_government_bonds():
    government_bonds = [
        {"face_value": 1, "interest_rate": 0.03, "max_bonds": 10000000000000},
        {"face_value": 1, "interest_rate": 0.02, "max_bonds": 50000000000000},
        {"face_value": 1400, "interest_rate": 0.02, "max_bonds": 5000},
        # Add more bonds with different interest rates and limits as needed
    ]
    return government_bonds
"""
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
"""
def calculate_annual_bond_return(remaining_investment):
    interest_rate = 0.03
    annual_bond_return = remaining_investment * interest_rate
    return annual_bond_return

def portfolio_value_at_risk(separate_portfolio_value,portfolio_years):
    csv_file_path1 = r'./dataset/EQNR_daily2022.csv'
    csv_file_path2 = r'./dataset/TRMED.OL.csv'
    csv_file_path3 = r'./dataset/TEL.OL.csv'
    close_price_list1 = read_csv_file_and_return_close_price_list(csv_file_path1)
    close_price_list2 = read_csv_file_and_return_close_price_list(csv_file_path2)
    close_price_list3 = read_csv_file_and_return_close_price_list(csv_file_path3)
    # Create DataFrames for the close price lists
    df1 = pd.DataFrame(close_price_list1, columns=['EQNR_Close'])
    df2 = pd.DataFrame(close_price_list2, columns=['TRMED_Close'])
    df3 = pd.DataFrame(close_price_list3, columns=['TEL_Close'])
    # Concatenate DataFrames horizontally
    close_df = pd.concat([df1, df2, df3], axis=1)
    stock_weights = np.array([0.4,0.2,0.4])
    # Calculate log returns
    log_returns = log_return(close_df)
    cov_matrix = log_returns.cov()*252
    z_score = random_z_score()
    portfolio_expected_return = expected_return(stock_weights, log_returns)
    portfolio_std_dev = standard_deviation(stock_weights, cov_matrix)
    
    #create a function to calculate scenarioGainloss
    days = portfolio_years #assuming 252 trading days
    confidence_interval = 0.95 #higher the confidence interval is the greater the value at risk will bc move further out the tail of the distribution
    simulations = 10000
    
    VaR,scenarioReturn = Value_at_Risk(simulations,separate_portfolio_value,portfolio_expected_return,portfolio_std_dev,z_score,days,confidence_interval)
    plot_histogram_VaR(VaR,scenarioReturn,portfolio_years, confidence_interval)
    return VaR

def compute_list_of_stock_returns_for_companies(portfolio_time, company1, company2,company3 ,wacc1, wacc2,wacc3,mean_factor,variance_factor):
    csv_file_path1 = r'./dataset/EQNR_daily2022.csv'
    csv_file_path2 = r'./dataset/TRMED.OL.csv'
    csv_file_path3 = r'./dataset/TEL.OL.csv'
    csv_file_path4 = r'./dataset/CRBP.csv'
    csv_file_path5 = r'./dataset/KOG.OL.csv'
    csv_file_path6 = r'./dataset/SNV.V.csv'
    csv_file_path7 = r'./dataset/CRBP.csv'
    csv_file_path8 = r'./dataset/SAS.ST.csv'
    csv_file_path9 = r'./dataset/ZAP.OL.csv'
    close_price_list1 = read_csv_file_and_return_close_price_list(csv_file_path1)
    close_price_list2 = read_csv_file_and_return_close_price_list(csv_file_path5)
    close_price_list3 = read_csv_file_and_return_close_price_list(csv_file_path6)
    
    #mean_factor = 1.0
    #variance_factor = 1.0
    simulated_prices1 = simulate_future_stock_prices(close_price_list1,mean_factor,variance_factor,portfolio_time)
    simulated_prices2 = simulate_future_stock_prices(close_price_list2,mean_factor,variance_factor,portfolio_time)
    simulated_prices3 = simulate_future_stock_prices(close_price_list3,mean_factor,variance_factor,portfolio_time)

    last_day_company1 = last_day_sim_price(simulated_prices1)
    last_day_company2 = last_day_sim_price(simulated_prices2)
    last_day_company3 = last_day_sim_price(simulated_prices3)
    
    stock_returns1 = []
    stock_returns2 = []
    stock_returns3 = []
    present_value_per_share1 = calculate_present_value_per_share(company1.future_value, wacc1, portfolio_time, company1.shares_available)
    present_value_per_share2 = calculate_present_value_per_share(company2.future_value, wacc2, portfolio_time, company2.shares_available)
    present_value_per_share3 = calculate_present_value_per_share(company3.future_value, wacc3, portfolio_time, company3.shares_available)
    #if present_value_per_share1 < company1.initial_price:
        #print(f"Dont buy it")
    
    #if present_value_per_share2 < company2.initial_price:
        #print(f"Dont buy it")
    
    #if present_value_per_share3 < company3.initial_price:
        #print(f"Dont buy it")
    # Calculate parametersv
    for year in range(1, portfolio_time):
        ### COMPANY 1 and 2 and 3
        stock_return1 = calculate_each_stock_return(last_day_company1[year], last_day_company1[year-1])
        stock_returns1.append(stock_return1)
        stock_return2 = calculate_each_stock_return(last_day_company2[year], last_day_company2[year-1])
        stock_returns2.append(stock_return2)
        stock_return3 = calculate_each_stock_return(last_day_company3[year], last_day_company3[year-1])
        stock_returns3.append(stock_return3)   
    
    stock_return_last = calculate_each_stock_return(last_day_company1[portfolio_time-1], last_day_company1[portfolio_time-2])
    stock_returns1.append(stock_return_last)
    
    stock_return_last2 = calculate_each_stock_return(last_day_company2[portfolio_time-1], last_day_company2[portfolio_time-2])
    stock_returns2.append(stock_return_last2)
    
    stock_return_last3 = calculate_each_stock_return(last_day_company3[portfolio_time-1], last_day_company3[portfolio_time-2])
    stock_returns3.append(stock_return_last3)
    
    return stock_returns1, stock_returns2,stock_returns3


def read_csv_file_and_return_close_price_list(csv_file_path):
    with open(csv_file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)
        close_price_list = []
        for row in csvreader:    
            close_price = float(row[4])            
            if close_price != 0:
                close_price_list.append(close_price)

        return close_price_list
    
def calculation_stock_return_for_historical_data(historical_stock_prices):
    stock_return = []
    for year in range(1,len(historical_stock_prices)):
        r1 = (historical_stock_prices[year] - historical_stock_prices[year-1])/historical_stock_prices[year-1]
        stock_return.append(r1)
    return stock_return

def simulate_future_stock_prices(close_price_list,mean_factor,variance_factor,num_years):
    #if seed is not None:
        #np.random.seed(seed)
    # Number of trading days in a year
    trading_days_per_year = 252
    
    # Initial prices for the first year
    simulated_prices_list = [close_price_list]

    # Simulate stock prices for each year
    for year in range(1, num_years+1):
        # Get the simulated prices of the previous year
        previous_year_prices = simulated_prices_list[-1]
            
        # Calculated mean and variance of returns for the current year
        stock_return_for_close_price = calculation_stock_return_for_historical_data(previous_year_prices)
        mean_return = np.mean(stock_return_for_close_price) * mean_factor
        variance_return = np.var(stock_return_for_close_price) * variance_factor
        
        # Simulate prices for the current year based on previous year prices
        simulated_prices = [previous_year_prices[-1]]  # Initial price for the year
        
        for _ in range(trading_days_per_year):
            # Generate a random daily return from a normal distribution
            daily_return = np.random.normal(mean_return, np.sqrt(variance_return))
            
            # Calculate the next day's stock price
            next_price = simulated_prices[-1] * (1 + daily_return)
        
            # Ensure next price is non-negative
            while next_price <= 0:
                daily_return = np.random.normal(mean_return, np.sqrt(variance_return))
                next_price = simulated_prices[-1] * (1 + daily_return)
              
            # Append the next day's price to the list
            simulated_prices.append(next_price)
        
        # Append the simulated prices for the year to the list of lists
        simulated_prices_list.append(simulated_prices)
    # Remove the first element from the list (simulated_prices_list[0])
    
    simulated_prices_list = simulated_prices_list[1:]
    return simulated_prices_list

def last_day_sim_price(simulated_prices_list_company):
    last_day_of_sim_price_company = []
    last_day_of_sim_price_company.append(simulated_prices_list_company[0][0])
    for simulated_prices in simulated_prices_list_company:
        last_day_price = simulated_prices[-1]
        last_day_of_sim_price_company.append(last_day_price)
    return last_day_of_sim_price_company

def generate_guarantee_portfolio(portfolio_time, initial_investment, guarantee_interest):
    guarante_portfolio = []
    for _ in range(portfolio_time):
        initial_investment = initial_investment*(1+guarantee_interest)
        guarante_portfolio.append(initial_investment)
    return guarante_portfolio


def portfolio_value_simulation(portfolio_time,initial_investment,pension_bond_weight,pension_stock_weight,guarantee_portfolio,buffer_bond_weight,buffer_stock_weight,mean_factor,variance_factor):
    expected_return_market_portfolio = 0.04
    bond_new_investment_separate = initial_investment * pension_bond_weight
    
    # Calculate Bond
    total_stock_investment_for_separate= initial_investment * pension_stock_weight
    stock_weights_allocation_for_all_companies = [0.4, 0.2, 0.4]
     
    stock_investments_all_companies = [total_stock_investment_for_separate* weight for weight in stock_weights_allocation_for_all_companies]
    separate_bond_return = calculate_annual_bond_return(bond_new_investment_separate)

    # Create Company
    company1 = Company(name="EQNR",future_value=3600000000,stock_beta=0.5,cost_of_debt=0.06,equity_weight=0.1,debt_weight=0.0,tax_rate=0.5,risk_free_rate=0.03,initial_price=35.81,shares_available=10000000)
    company2 = Company(name="TRMED",future_value=10000000,stock_beta=0.5,cost_of_debt=0.06,equity_weight=0.2,debt_weight=0.0,tax_rate=0.5,risk_free_rate=0.03,initial_price=1.105, shares_available=1000000)
    company3 = Company(name="TEL",future_value=1170000000,stock_beta=0.5,cost_of_debt=0.01,equity_weight=0.3,debt_weight=0.02,tax_rate=0.4,risk_free_rate=0.03,initial_price=115.52, shares_available=1000000)
    
    ### COMPANY 1
    cost_of_equity1 = calculate_cost_of_equity(company1.risk_free_rate, company1.stock_beta, expected_return_market_portfolio)
    wacc1 = calculate_wacc(cost_of_equity1, company1.cost_of_debt, company1.equity_weight, company1.debt_weight, company1.tax_rate)
    ### COMPANY 2
    cost_of_equity2 = calculate_cost_of_equity(company2.risk_free_rate, company2.stock_beta, expected_return_market_portfolio)
    wacc2 = calculate_wacc(cost_of_equity2, company2.cost_of_debt, company2.equity_weight, company2.debt_weight, company2.tax_rate)
    ### COMPANY 3
    cost_of_equity3 = calculate_cost_of_equity(company3.risk_free_rate, company3.stock_beta, expected_return_market_portfolio)
    wacc3 = calculate_wacc(cost_of_equity3, company3.cost_of_debt, company3.equity_weight, company3.debt_weight, company3.tax_rate)
    ### list of stocks returns
    stock_returns1, stock_returns2,stock_returns3 = compute_list_of_stock_returns_for_companies(portfolio_time, 
                                                                   company1, company2,company3, 
                                                                   wacc1, wacc2,wacc3,mean_factor,variance_factor)
    stock_returns_all_companies = [stock_returns1, stock_returns2,stock_returns3]
    
    
    separate_portfolio_account = Portfolio("Portfolio_Separate", initial_investment,pension_bond_weight, pension_stock_weight)
    buffer_separate_account = Buffer("Buffer_Separate", 0)
    equity_separate_account = Equity("Equity_Separate", 0)
    separate_portfolio_return = 0
    #Accounts for single portfolio
    single_portfolio_account = Portfolio("Portfolio_single_pensionbuffer", initial_investment, pension_bond_weight, pension_stock_weight)
    equity_single_account = Equity("Equity_single_pensionbuffer", 0)
    equity_single_account_list = []
    equity_single_account_list.append(0)
    
    portfolio_value_separate_list = []
    #portfolio_with_separate_buffer_list is the separate portfolios 
    portfolio_with_separate_buffer_list = []
    portfolio_value_separate_list.append(initial_investment)
    portfolio_with_separate_buffer_list.append(initial_investment)
    #### Istedenfor single skrive Single på alle####
    #portfolio_single_investment_list is the single investment
    portfolio_single_investment_list = []
    portfolio_single_investment_list.append(initial_investment)
    portfolio_value_allocated_list = []
    buffer_investment_money_list = []
    buffer_investment_money_list.append(0)
    equity_separate_account_list = []
    equity_separate_account_list.append(0)
    separate_portfolio_value = initial_investment
    separate_stock_return = calculate_total_stock_return(0, stock_investments_all_companies, stock_returns_all_companies, portfolio_time)

    #Single portfolio Investment
    single_portfolio_value = initial_investment
    single_portfolio_return = 0
    single_bond_return = calculate_annual_bond_return(bond_new_investment_separate)
    total_stock_investment_single = initial_investment * pension_stock_weight
    bond_new_investment_for_single = initial_investment *pension_bond_weight
    portfolio_single_investment_list = []
    portfolio_single_investment_list.append(initial_investment)
    #single return first year
    single_stock_return = separate_stock_return
    single_bond_return = separate_bond_return
    
    
    for year in range(portfolio_time): 
        #Two Separate Portfolios
        separate_portfolio_return = (separate_stock_return + separate_bond_return)
        separate_portfolio_value = separate_portfolio_value + separate_portfolio_return
        bond_new_investment_separate += separate_bond_return          

        total_stock_investment_for_separate += separate_stock_return
        portfolio_value_separate_list.append(separate_portfolio_value)
        portfolio_with_separate_buffer_list.append(separate_portfolio_value + buffer_separate_account.balance)
            
        #Pension and buffer in Single Portfolio
        single_portfolio_return = (single_stock_return + single_bond_return)
        single_portfolio_value = single_portfolio_value + single_portfolio_return
        bond_new_investment_for_single += single_bond_return 
        total_stock_investment_single += single_stock_return
        portfolio_single_investment_list.append(single_portfolio_value)
        """    
        #print(f"Before calculation: Year {year + 1}: Portfolio variance: {portfolio_variance}")
        print(f"Before calculation: Year {year + 1}: Bond Value: {bond_new_investment_separate:.2f}, Bond return: {separate_bond_return:.2f}")
        print(f"Before calculation: Year {year + 1}: Stock Value: {total_stock_investment:.2f}, Stock Return: {stock_return:.2f}")
        print(f"Before calculation: Year {year + 1}: Portfolio Value: {separate_portfolio_value:.2f}, Portfolio Return: {separate_portfolio_return:.2f}")
        print("----------------------------------------------------------------------------------------------")
        """
        bond_guarantee_value = guarantee_portfolio[year] * separate_portfolio_account.bond_weight
        stock_guarantee_value = guarantee_portfolio[year] * separate_portfolio_account.stock_weight
        
        #Reallocation of bonds and stocks for separated portfolio #NEW CHANGE
        bond_allocation = bond_new_investment_separate - bond_guarantee_value
        stock_allocation = total_stock_investment_for_separate- stock_guarantee_value
        portfolio_allocation = stock_allocation + bond_allocation
        bond_new_investment_separate = guarantee_portfolio[year] * separate_portfolio_account.bond_weight
        stock_guaranteed_new_investments_all_companies = [stock_guarantee_value * weight for weight in stock_weights_allocation_for_all_companies]
        separate_stock_return = calculate_total_stock_return(year, stock_guaranteed_new_investments_all_companies, stock_returns_all_companies, portfolio_time) 
        
        #NEW CHANGE #Tranfer between 3 accounts using reallocation logic 
        total_stock_investment_for_separate= stock_guarantee_value 
        separate_bond_return = calculate_annual_bond_return(bond_new_investment_separate) 
        if separate_portfolio_value > guarantee_portfolio[year]:
            surplus_separate_portfolio = abs(separate_portfolio_value - guarantee_portfolio[year])         
            buffer_separate_account.deposit(surplus_separate_portfolio,year)
            
        elif separate_portfolio_value < guarantee_portfolio[year]:
            deficit_separate_portfolio = abs(separate_portfolio_value - guarantee_portfolio[year])
            if buffer_separate_account.balance > 0:
                if buffer_separate_account.balance >= deficit_separate_portfolio:
                    buffer_separate_account.withdraw(deficit_separate_portfolio,year)
                    deficit_separate_portfolio = 0
                else:
                    deficit_separate_portfolio -= buffer_separate_account.balance
                    buffer_separate_account.withdraw(buffer_separate_account.balance,year)

            equity_separate_account.withdraw(deficit_separate_portfolio,year)
        #Reallocation of bonds and stocks for single portfolio #NEW CHANGE
        bond_single_allocation = bond_new_investment_for_single - bond_guarantee_value
        stock_single_allocation = total_stock_investment_single - stock_guarantee_value
        portfolio_single_allocation = stock_single_allocation + bond_single_allocation
        
        if single_portfolio_value > guarantee_portfolio[year]:
            buffer_single_investment = single_portfolio_value - guarantee_portfolio[year]
        else:
            buffer_single_investment= 0
        #Deficit 
        if  single_portfolio_value < guarantee_portfolio[year]:
            deficit_single_portfolio = abs(guarantee_portfolio[year] - single_portfolio_value)
            equity_single_account.withdraw(deficit_single_portfolio,year)
            deficit_single_portfolio = 0
            single_portfolio_value = guarantee_portfolio[year]
        
        bond_new_investment_for_single = single_portfolio_value * single_portfolio_account.bond_weight
        stock_guarantee_buffer_value = single_portfolio_value * single_portfolio_account.stock_weight
        
        stock_new_investments_buffer_per_company = [stock_guarantee_buffer_value * weight for weight in stock_weights_allocation_for_all_companies]
        single_stock_return = calculate_total_stock_return(year, stock_new_investments_buffer_per_company,stock_returns_all_companies,portfolio_time)
        #NEW CHANGE #Uptade the stock investment
        total_stock_investment_single = stock_guarantee_buffer_value 
        single_bond_return = calculate_annual_bond_return(bond_new_investment_for_single) 
    
        single_portfolio_account.deposit(guarantee_portfolio[year]- single_portfolio_account.balance, year)
        portfolio_value_allocated_list.append(portfolio_single_allocation)
        separate_portfolio_account.deposit(guarantee_portfolio[year] - separate_portfolio_account.balance, year)
        
        separate_portfolio_value = separate_portfolio_account.balance

        portfolio_value_allocated_list.append(portfolio_allocation)
        """
        print(f"After allocation: Year {year + 1}: Bond Value: {bond_guarantee_value:.2f}, Allocation of bond return: {bond_allocation:.2f}")
        print(f"After allocation: Year {year + 1}: Stock Value: {stock_guarantee_value:.2f}, Allocation of stock Return: {stock_allocation:.2f}")
        print(f"After allocation: Year {year + 1}: Portfolio Value: {separate_portfolio_value:.2f}, Allocation of portfolio Return: {portfolio_allocation:.2f}")
        print("----------------------------------------------------------------------------------------------")
        """
        equity_separate_account_list.append(equity_separate_account.balance)
        equity_single_account_list.append(equity_single_account.balance)
        equity_separate_portfolio_balance_CF = calculate_discounted_cashflow(equity_separate_account.balance,0.05,portfolio_time)
        equity_single_portfolio_balance_CF = calculate_discounted_cashflow(equity_single_account.balance,0.05,portfolio_time)
        buffer_separate_investment_CF = calculate_discounted_cashflow(buffer_separate_account.balance,0.05,portfolio_time)  
        buffer_investment_money = buffer_value_simulation(year,buffer_separate_account.balance,buffer_bond_weight,buffer_stock_weight, stock_returns_all_companies,portfolio_time)
        buffer_separate_account.balance = buffer_investment_money
        buffer_investment_money_list.append(buffer_investment_money)
        buffer_single_investment_CF = calculate_discounted_cashflow(buffer_single_investment,0.05,portfolio_time)
        #print(f"After allocation: Year {year + 1}: Buffer balance Value: {buffer_separate_account.balance:.2f}, Buffer money: {buffer_investment_money:.2f}")
        
        #print(f"Discounted Equity used after {portfolio_time} years is : {equity_balance_CF:.2f}")
        #print(f"Discounted total amount saved in Buffer after {portfolio_time} years is : {buffer_balance_CF:.2f}")
    return portfolio_with_separate_buffer_list, portfolio_single_investment_list, equity_separate_portfolio_balance_CF, buffer_separate_investment_CF,buffer_single_investment_CF,equity_single_portfolio_balance_CF

def buffer_value_simulation(year,initial_investment, buffer_bond_weight, buffer_stock_weight, stock_returns_per_company,portfolio_time):
    
    bond_new_investment_separate = initial_investment * buffer_bond_weight
    
    # Calculate Bond
    total_stock_investment_for_separate = initial_investment * buffer_stock_weight
    buffer_stock_weight_list = [0.2, 0.4, 0.4]
    
    stock_investments_per_company = [total_stock_investment_for_separate * weight for weight in buffer_stock_weight_list]
    separate_bond_return = calculate_annual_bond_return(bond_new_investment_separate)

    separate_portfolio_return = 0
    separate_portfolio_value = initial_investment
    stock_return = calculate_total_stock_return(year, stock_investments_per_company, stock_returns_per_company, portfolio_time)
    separate_portfolio_return = (stock_return + separate_bond_return)
    separate_portfolio_value = separate_portfolio_value + separate_portfolio_return

    return separate_portfolio_value
def print_stats(list, account,simulation_portfolio):
    mean_list = np.mean(list)
    array_list= np.array(list)
    variance_list = np.var(list)
    max_list = max(list)
    min_list = min(list)
    quantile_1 = np.percentile(list,1)
    quantile_5 = np.percentile(list,5)
    count = list.count(0)
    freq = count/simulation_portfolio
    print(f"{account} : Mean = {mean_list}, Var= {variance_list}, Max = {max_list}, Min = {min_list}, 1% Quantile: {quantile_1}, 5% Quantile: {quantile_5}, frequency: {freq}")
    return f"{account} : Mean = {mean_list}, Var= {variance_list}, Max = {max_list}, Min = {min_list}, 1% Quantile: {quantile_1}, 5% Quantile: {quantile_5}, frequency: {freq}\n"


def main():
    print("----------------------------------------------------------------------------------------------")
    #list_pension_bond_weight = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    #list_buffer_bond_weight = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    #husk kjor de imorgen::::0.5,0.6,1.0
    list_pension_bond_weight = [0.7,0.5]
    list_buffer_bond_weight = [0.0,0.5,0.9,1.0]
    
    for aa in range(len(list_pension_bond_weight)):
        for bb in range(len(list_buffer_bond_weight)): 
    
            startTime = dt.datetime.now()
            #portfolio_time = int(input("Enter the number of years of portfolio: "))
            portfolio_time = 20
            initial_investment = 100
            guarantee_portfolio = generate_guarantee_portfolio(portfolio_time, initial_investment, 0.04)
            #print(sum(guarantee_portfolio))
            simulation_portfolio = 100
            equity_separate_portfolio_list = []
            equity_single_portfolio_list = []
            buffer_separate_list = []
            portfolio_pension_buffer_list = []
            portfolio_values_separate_list = []
            buffer_single_list = []
            #Pension Weights
            pension_bond_weight = list_pension_bond_weight[aa]
            pension_stock_weight = 1 - pension_bond_weight
            #Buffer Weights
            buffer_bond_weight = list_buffer_bond_weight[bb]
            buffer_stock_weight = 1 - buffer_bond_weight
            
            for i in range(simulation_portfolio):
                #print(f"year sim :{i}")
                portfolio_separate_values,portfolio_pension_buffer,equity_separate,buffer_separate,buffer_single,equity_single_account = portfolio_value_simulation(portfolio_time,initial_investment,pension_bond_weight,pension_stock_weight,guarantee_portfolio, buffer_bond_weight,buffer_stock_weight,0.6,0.6)
                portfolio_values_separate_list.append(portfolio_separate_values)
                portfolio_pension_buffer_list.append(portfolio_pension_buffer)
                equity_separate_portfolio_list.append(equity_separate)
                equity_single_portfolio_list.append(equity_single_account)
                buffer_separate_list.append(buffer_separate)
                buffer_single_list.append(buffer_single)
                #print("----------------------------------------------------------------------------------------------")
            
            """
            m_factor = [0.75,1.0,1.1]
            v_factor = [0.75,1.0,1.1]
            name_list_factor = []
            list_mean_var_adjust = []
            for i in range(len(m_factor)):
                for k in range(len(v_factor)):
                    portfolio_separate_values,portfolio_pension_buffer,equity_separate,buffer_separate,buffer_single,equity_single_account_list = portfolio_value_simulation(portfolio_time,initial_investment,bond_weight,stock_weight,guarantee_portfolio, buffer_bond_weight,buffer_stock_weight,m_factor[i],v_factor[k])
                    list_mean_var_adjust.append(portfolio_separate_values)
                    name_list_factor.append(f'Factor Adjustment: ({m_factor[i]},{v_factor[k]})')
            endTime = dt.datetime.now()
            timeUsed = endTime - startTime
            print(f"Time used: {timeUsed.total_seconds()} sec")
            #print(f"Time used: {timeUsed} minutes")
            guarantee_portfolio.insert(0,initial_investment)
            print(list_mean_var_adjust)
            factor_plot = plot_mv_factor_graphs(portfolio_time,9,list_mean_var_adjust,name_list_factor)
            """
            f = open(f"results/extra_results_{simulation_portfolio}_Bond_weight={buffer_bond_weight}_pension_bond_weight={pension_bond_weight}.txt", "w")
            endTime = dt.datetime.now()
            timeUsed = endTime - startTime
            print(f"Time used: {timeUsed.total_seconds()} sec")
            #print(f"Time used: {timeUsed} minutes")

            f.write(f"Time used: {timeUsed.total_seconds()} sec")
            f.write(print_stats(equity_separate_portfolio_list, 'Equity Separate Portfolios',simulation_portfolio))
            f.write(print_stats(equity_single_portfolio_list, 'Equity Single Portfolio',simulation_portfolio))
            f.write(print_stats(buffer_separate_list, 'Buffer Separate Portfolios',simulation_portfolio))
            f.write(print_stats(buffer_single_list, 'Buffer Single Portfolio',simulation_portfolio))
            f.close()    
            # Plot the first histogram
            b = plot_equity_histogram(equity_separate_portfolio_list,equity_single_portfolio_list, 'Equity Separate Portfolios','Equity Single Portfolio', simulation_portfolio, buffer_bond_weight, pension_bond_weight)
            d = plot_buffer_histogram(buffer_separate_list,buffer_single_list, 'Buffer Separate Portfolio','Buffer Single Portfolio', simulation_portfolio, buffer_bond_weight, pension_bond_weight)
            #j = plot_outliers_histogram(buffer_separate_list,buffer_single_list, 'Buffer Separate Portfolio','Buffer Single Portfolio', simulation_portfolio, buffer_bond_weight, pension_bond_weight)
        #plt.show() #REMEMBER: JEG HAR KOMMERTE DENNE BORT for testing;
            #c = plot_buffer_histogram(buffer_list, 'Buffer')
            # Adjust layout to prevent overlapping
            #plt.tight_layout()
            
            """
            fig, (ax5, ax6) = plt.subplots(2, 1, figsize=(8, 6))
            g = plot_outliers_histogram(ax5,buffer_list, 'Buffer Separate Portfolio')
            h = plot_outliers_histogram(ax6,buffer_investment_money_list, 'Buffer single Portfolio')
            #e = plot_buffer_histogram(ax4,buffer_single_list, 'Buffer single Portfolio')
            plt.show()
            """
            #plt.show()
            #d = portfolio_value_at_risk(initial_investment,portfolio_time)

            #guarantee_portfolio.insert(0,initial_investment)
            #list_number = simulation_portfolio-1
            #a = plot_two_graphs_difference(portfolio_time,portfolio_values_separate_list[list_number], portfolio_pension_buffer_list[list_number],guarantee_portfolio, buffer_separate_list[list_number], equity_separate_portfolio_list[list_number])
            #plt.show()
            
    
            #plt.subplot(2, 2,1)
            #fig, (ax1, ax1) = plt.subplots(2, 1, figsize=(8, 6))
            #c = plot_histogram2(ax1,equity_single_portfolio_list, 'Equity single Portfolio')
            #fig, (ax3, ax3) = plt.subplots(2, 1, figsize=(8, 6))
if __name__ == "__main__":
    main()
