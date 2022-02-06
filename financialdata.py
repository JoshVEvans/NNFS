import pandas as pd


csv = pd.read_csv("financial_data_sp500_companies_1.csv")
xlsx = pd.read_excel("financial_data_sp500_companies_final.xlsx")


print(csv.isna().sum())


csv = csv.dropna()

print(csv.isna().sum())
