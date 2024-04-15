
def calculate_cost_of_equity(risk_free_rate, beta, expected_return):
    return risk_free_rate + beta * (expected_return - risk_free_rate)

def calculate_wacc(cost_of_equity, cost_of_debt, equity_weight, debt_weight, tax_rate):
    return (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))

def calculate_present_value_per_share(future_value, wacc, time, shares_available):
    return (future_value / (1 + wacc) ** time) / shares_available

def calculate_discounted_cashflow(balance, interest_rate, years):
    return balance/((1+ interest_rate)**years)

