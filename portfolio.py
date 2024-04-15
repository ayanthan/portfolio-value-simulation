from account import Account

class Portfolio(Account):
    def __init__(self, account_number, balance, bond_weight,stock_weight):
        super().__init__(account_number, balance, False)
        self.stock_weight = stock_weight
        self.bond_weight = bond_weight

class Buffer(Account):
    def __init__(self, account_number, balance):
        super().__init__(account_number, balance, False)
    

class Equity(Account):
    def __init__(self, account_number, balance):
        super().__init__(account_number, balance, True)