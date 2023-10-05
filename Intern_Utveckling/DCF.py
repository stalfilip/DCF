import yfinance as yf
import numpy as np
import pandas as pd

class DCF:

    def __init__(self, uncer_in_fcf, outstanding_shares, path_to_company_data, nr_forecast_years, market_return, cost_of_debt, tax_rate, perpetual_growth_rate, stock_yf_code, market_yf_code, beta = None):

        self.df = pd.read_excel(path_to_company_data, sheet_name='Year', engine='openpyxl')
        self.uncertainty_in_fcf = uncer_in_fcf
        self.avg_fcf_growth = self.avg_fcf_growth(self.df)
        self.latest_fcf = self.get_latest_fcf(self.df)
        self.lower_cash_flow, self.cash_flows,  self.upper_cash_flow = self.forecast_fcf(self.latest_fcf, nr_forecast_years, self.avg_fcf_growth)
        self.debt_weight, self.equity_weight = self.calculate_weights(self.df)
        self.stock_yf_code = stock_yf_code  # Flyttad före beta-beräkningen
        self.market_yf_code = market_yf_code  # Flyttad före beta-beräkningen
        self.cost_of_debt = cost_of_debt
        self.outstanding_shares = outstanding_shares
        if beta is None:
            self.beta = self.calculate_beta(cost_of_debt)
        else:
            self.beta = beta
        self.market_return = market_return
        self.tax_rate = tax_rate
        self.perpetual_growth_rate = perpetual_growth_rate
        self.cost_of_equity = self.calculate_cost_of_equity()
        self.discount_rate = self.calculate_wacc()

    def get_outstanding_shares(self):
        stock = yf.Ticker(self.stock_yf_code)
        outstanding_shares = stock.info['sharesOutstanding']
        #print(f"Outstanding shares: {outstanding_shares}")
        return outstanding_shares * 10**(-6)

    def get_latest_fcf(self, df):
        fcf_index = df[df.iloc[:, 0] == 'Free Cash Flow'].index[0]
        latest_year = df.columns[-1]  # Hämtar den sista kolumnen, vilket är det senaste året.
        latest_fcf = df.at[fcf_index, latest_year]
        #print(latest_fcf)
        return latest_fcf

    def avg_fcf_growth(self, df):
        growth_index = df[df.iloc[:, 0] == 'FCF growth'].index[0]
        latest_years = df.columns[-3:]  # Hämtar de tre sista kolumnerna.
        avg_fcf_growth = df.loc[growth_index, latest_years].mean()
        #print(f"avg_fcf_growth: {avg_fcf_growth}")
        maximum_growth = 3
        return min(avg_fcf_growth, maximum_growth)
    
    def forecast_fcf(self, latest_fcf, nr_forecast_years, avg_fcf_growth):
        lower_future_cash_flows = [latest_fcf]
        future_cash_flows = [latest_fcf]  
        upper_future_cash_flows = [latest_fcf]
        #print(f"avg_fcf_growth: {avg_fcf_growth}")
        lower_growth = avg_fcf_growth * (1 - self.uncertainty_in_fcf)
        upper_growth = avg_fcf_growth * (1 + self.uncertainty_in_fcf)

        for _ in range(nr_forecast_years):
            next_fcf = future_cash_flows[-1] * (1 + avg_fcf_growth/100)
            lower_next_fcf = lower_future_cash_flows[-1] * (1 + lower_growth/100)
            upper_next_fcf = upper_future_cash_flows[-1] * (1 + upper_growth/100)
            future_cash_flows.append(next_fcf)
            lower_future_cash_flows.append(lower_next_fcf)
            upper_future_cash_flows.append(upper_next_fcf)

        return lower_future_cash_flows, future_cash_flows,  upper_future_cash_flows

    def calculate_weights(self, df):
        latest_year = df.columns[-1]
        total_equity_row = df[df["Report"] == "Total Equity"]
        total_equity = total_equity_row[latest_year].iloc[0]
        total_liabilities_row = df[df["Report"] == "Total liabilities"]
        total_liabilities = total_liabilities_row[latest_year].iloc[0]
        debt_weight = total_liabilities / (total_liabilities + total_equity)
        equity_weight = 1 - debt_weight
        #print(f"Debt weight: {debt_weight}")
        #print(f"Equity weight: {equity_weight}")
        return debt_weight, equity_weight
    
    def calculate_beta(self, risk_free_rate=0.01, start_date='2021-01-01', end_date='2023-01-01'):

        stock_data = yf.download(self.stock_yf_code, start=start_date, end=end_date, progress = False)
        market_data = yf.download(self.market_yf_code, start=start_date, end=end_date, progress = False)
        common_dates = stock_data.index.intersection(market_data.index)
        stock_prices = stock_data.loc[common_dates]['Close'].tolist()
        market_prices = market_data.loc[common_dates]['Close'].tolist()
        stock_returns = [np.log(stock_prices[i] / stock_prices[i-1]) for i in range(1, len(stock_prices))]
        market_returns = [np.log(market_prices[i] / market_prices[i-1]) for i in range(1, len(market_prices))]
        adjusted_stock_returns = [ret - risk_free_rate for ret in stock_returns]
        adjusted_market_returns = [ret - risk_free_rate for ret in market_returns]
        beta = np.cov(adjusted_stock_returns, adjusted_market_returns)[0][1] / np.var(adjusted_market_returns)
        return beta        
               
    def calculate_cost_of_equity(self):
        """
        Beräknar kostnaden för eget kapital med hjälp av CAPM.
        """

        return self.cost_of_debt + self.beta * (self.market_return - self.cost_of_debt)

    def calculate_wacc(self):
        """
        Beräknar Weighted Average Cost of Capital (WACC).
        """
        wacc = (self.equity_weight * self.cost_of_equity) + (self.debt_weight * self.cost_of_debt * (1 - self.tax_rate))
       
        return wacc

    def calculate_present_value(self):
        """
        Beräknar nuvärdet av en serie kassaflöden givet en diskonteringsränta.
        """
        present_value = self.cash_flows[0]
        lower_present_value = self.lower_cash_flow[0]
        upper_present_value = self.upper_cash_flow[0]

        for i, cash_flow in enumerate(self.cash_flows[1:]):  
            present_value += cash_flow / (1 + self.discount_rate)**(i+1)

        for i, cash_flow in enumerate(self.lower_cash_flow[1:]):  
            lower_present_value += cash_flow / (1 + self.discount_rate)**(i+1)

        for i, cash_flow in enumerate(self.upper_cash_flow[1:]):
            upper_present_value += cash_flow / (1 + self.discount_rate)**(i+1)
            
        return lower_present_value, present_value, upper_present_value

    def calculate_terminal_value(self):
        """
        Beräknar det eviga värdet av kassaflöden med en konstant tillväxttakt.

        """
        lower_terminal_value = self.lower_cash_flow[-1] * (1 + self.perpetual_growth_rate) / (self.discount_rate - self.perpetual_growth_rate)
        terminal_value = self.cash_flows[-1] * (1 + self.perpetual_growth_rate) / (self.discount_rate - self.perpetual_growth_rate)
        upper_terminal_value = self.upper_cash_flow[-1] * (1 + self.perpetual_growth_rate) / (self.discount_rate - self.perpetual_growth_rate)
        #print(f"Terminal value: {terminal_value}")
        return lower_terminal_value, terminal_value, upper_terminal_value

    def perform_dcf(self):
        """
        Utför en DCF-analys.
        """
        lower_pv_of_cash_flows, pv_of_cash_flows, upper_pv_of_cash_flows = self.calculate_present_value()
        lower_terminal_value, terminal_value, upper_terminal_value = self.calculate_terminal_value()
        lower_total_value = lower_pv_of_cash_flows + lower_terminal_value
        total_value = pv_of_cash_flows + terminal_value
        upper_total_value = upper_pv_of_cash_flows + upper_terminal_value
        #print(f"Terminal Value / Total Value: {pv_of_terminal_value / total_value}")
        lower_valuation = lower_total_value / self.outstanding_shares
        valuation = total_value / self.outstanding_shares
        upper_valuation = upper_total_value / self.outstanding_shares
        
        return lower_valuation, valuation, upper_valuation


