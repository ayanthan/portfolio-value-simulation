class Account:
    def __init__(self, account_number, balance, allow_negative):
        self.account_number = account_number
        self.balance = balance
        self.allow_negative = allow_negative
        
    def deposit(self, amount,year):
        self.balance += amount
        #print(f"Deposit {self.balance} in {year} in account number {self.account_number}.")
        
    def withdraw(self, amount, year):
        if self.balance < amount and self.allow_negative == False:
            print("The amount is higher than the balance in account number {self.account_number}")
        else:
            self.balance -= amount
        #print(f"Withdraw {self.balance} in {year} in account number {self.account_number}.")
        
    def display_balance(self):
        print(f"Account {self.account_number} has a balance of ${self.balance:.2f} in account number {self.account_number}")
