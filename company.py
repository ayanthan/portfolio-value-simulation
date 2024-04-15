class Company:
    def __init__(self, name, future_value, stock_beta, cost_of_debt, equity_weight, debt_weight, tax_rate, risk_free_rate, initial_price, shares_available):
        self.name = name
        self.future_value = future_value
        self.stock_beta = stock_beta
        self.cost_of_debt = cost_of_debt
        self.equity_weight = equity_weight
        self.debt_weight = debt_weight
        self.tax_rate = tax_rate
        self.risk_free_rate = risk_free_rate
        self.initial_price = initial_price
        self.shares_available = shares_available