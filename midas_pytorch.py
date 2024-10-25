# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 19:39:34 2024

@author: JMSA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




df = pd.read_csv('LoansDatasest.csv')

# Converting relevant columns from object to numeric
df['customer_income']=df['customer_income'].str.replace(',','').str.replace('£','')
df['customer_income'] = pd.to_numeric(df['customer_income'], errors='coerce')

df['loan_amnt']=df['loan_amnt'].str.replace(',','').str.replace('£','')
df['loan_amnt'] = pd.to_numeric(df['loan_amnt'], errors='coerce')

# Plotting the distributions of the object type columns
sns.histplot(data=df[df['customer_income']<=200000],x='customer_income', hue='home_ownership') #, multiple='dodge', shrink=3)
plt.title("Customer Income Distribution")
plt.xlabel("Customer Income")
plt.ylabel("Count")
plt.show()

sns.histplot(data=df[df['customer_income']<=200000],x='customer_income', hue='loan_intent') #, multiple='dodge', shrink=3)
plt.title("Customer Income Distribution")
plt.xlabel("Customer Income")
plt.ylabel("Count")
plt.show()

sns.histplot(data=df[df['loan_amnt']<=200000],x='loan_amnt', hue='loan_intent') #, multiple='dodge', shrink=3)
plt.title("Loan amount distribution")
plt.xlabel("Loan amount")
plt.ylabel("Count")
plt.show()

sns.histplot(data=df[df['loan_amnt']<=200000],x='loan_amnt', hue='home_ownership') #, multiple='dodge', shrink=3)
plt.title("Loan amount distribution")
plt.xlabel("Loan amount")
plt.ylabel("Count")
plt.show()

