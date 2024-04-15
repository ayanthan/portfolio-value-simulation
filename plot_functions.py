import matplotlib.pyplot as plt
import numpy as np

def plot_two_graphs_difference(portfolio_time, portfolio_values_list, portfolio_pension_buffer_list,guarantee_portfolio, buffer_investment_money, equity_years_list):
    time = list(range(0, portfolio_time+1))
    #time = list(range(0, portfolio_time+1))
    plt.figure(figsize=(10,8))
    plt.plot(time, portfolio_values_list, color='blue', linestyle='-', label='Separate Portfolios')
    plt.plot(time, portfolio_pension_buffer_list, color='cyan', linestyle='-', label='Single Portfolio')
    plt.plot(time, guarantee_portfolio, color='black', linestyle='--', label='Guarantee portfolio')
    #plt.plot(time, buffer_investment_money, color='green', linestyle='-', label='Buffer value')
    #plt.plot(time, equity_years_list, color='red', linestyle='-', label='Equity value')
    difference = np.array(guarantee_portfolio) - np.array(portfolio_values_list)
    difference_pension_buffer = np.array(portfolio_pension_buffer_list) - np.array(portfolio_values_list)
    plt.fill_between(time, guarantee_portfolio, portfolio_values_list, where=(difference > 0), interpolate=True, color='red', alpha=0.7,label='Less than interest guarantee')
    plt.fill_between(time, guarantee_portfolio, portfolio_values_list, where=(difference <= 0), interpolate=True, color='green', alpha=0.7,label='More than interest guarantee')    
    plt.fill_between(time, portfolio_pension_buffer_list, portfolio_values_list, where=(difference_pension_buffer > 0), interpolate=True, color='purple', alpha=0.5,label='Separate Portfolios lower')
    plt.fill_between(time, portfolio_pension_buffer_list, portfolio_values_list, where=(difference_pension_buffer <= 0), interpolate=True, color='yellow', alpha=0.7,label='Separate Portfolios higher')
    #plt.fill_between(time2, buffer_investment_money, portfolio_values_list, where=(diff_buffer = 0), interpolate=True, color='purple', alpha=0.5)    
    plt.xticks(time)  # Set x-axis ticks to match the time values
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Portfolio value', fontsize=12)
    plt.title('Single Portfolio vs Separate Portfolios', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.show()

def plot_buffer_graphs(buffer_investment_money_list,portfolio_time):
    time2 = list(range(0, portfolio_time+1))
    for i in range(len(buffer_investment_money_list)):
        plt.plot(time2, buffer_investment_money_list[i], linestyle='-', label='Buffer value')
    plt.xticks(time2)    
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Portfolio value', fontsize=12)
    plt.title('Buffer Portfolio value', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.show()

    
def plot_mv_factor_graphs(portfolio_time,mean_factor, portfolio_values_list,name):
    time = list(range(0, portfolio_time+1))
    plt.figure(figsize=(10,8))
    for i in range((mean_factor)):    
        plt.plot(time, portfolio_values_list[i], linestyle='-',label=name[i])
    plt.xticks(time)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Portfolio values', fontsize=12)
    plt.title('Portfolio value Through Market Events', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.show()
    

def plot_equity_histogram(data, data2, name1,name2,simulation_portfolio, buffer_bond_weight, pension_bond_weight):
    plt.figure(num=f"Equity={simulation_portfolio}_Buffer_Bond_weight={buffer_bond_weight}_pension_bond_weight={pension_bond_weight}")
    plt.hist(data, color='red', label=name1, bins=30, density=True, alpha=1, linewidth=2)
    plt.hist(data2, color='purple', label=name2, bins=30, density=True, alpha=0.7, linewidth=2)
    plt.xlabel('Balance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Equity Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig(f"results/extra_Equity={simulation_portfolio}_Bond_weight={buffer_bond_weight}_pension_bond_weight={pension_bond_weight}.png", bbox_inches='tight')
    #plt.show()
    
def plot_buffer_histogram(data_list1, data_list2, name_list1,name_list2, simulation_portfolio, buffer_bond_weight, pension_bond_weight):
    plt.figure(num=f"Buffer={simulation_portfolio}_Buffer_Bond_weight={buffer_bond_weight}_pension_bond_weight={pension_bond_weight}")
    plt.hist(data_list1, color='green', label=name_list1, bins=30, range=(0, 10000), density=True, alpha =0.8)
    plt.hist(data_list2, color='cyan', label=name_list2, bins=30, range=(0, 10000), density=True, alpha =0.5)
    plt.xlabel('Balance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Buffer Surplus', fontsize=14)
    plt.xlim(0,10000)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig(f"results/extra_Buffer={simulation_portfolio}_Buffer_Bond_weight={buffer_bond_weight}_pension_bond_weight={pension_bond_weight}.png", bbox_inches='tight')
    #plt.show()
        
def plot_outliers_histogram(data_list1, data_list2,name_list1,name_list2, simulation_portfolio, buffer_bond_weight, pension_bond_weight, x_range=(10000, 1000000), bins=30):
    outliers1 = [value for value in data_list1 if (x_range[0] <= value <= x_range[1])]
    outliers2 = [value for value in data_list2 if (x_range[0] <= value <= x_range[1])]
    plt.hist(outliers1, color='green', label=name_list1 + ' Outliers', bins=bins, range=x_range, alpha=0.8, density=True)
    plt.hist(outliers2, color='cyan', label=name_list2 + ' Outliers', bins=bins, range=x_range, alpha=0.5, density=True)
    plt.xlabel('Balance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)  
    plt.title('Outliers from Buffer Accounts', fontsize=14)
    plt.xlim(10000,1000000)
    plt.legend(fontsize=10)
    plt.grid(True)
    #plt.savefig(f"results/extra_Outliers={simulation_portfolio}_Buffer_Bond_weight={buffer_bond_weight}_pension_bond_weight={pension_bond_weight}.png", bbox_inches='tight')
    #plt.show()

