# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 19:39:34 2024

@author: JMSA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F


class classifierNN(nn.Module):
    def __init__(self, insize, outsize, nn_hlayer, p=0):
        super(classifierNN, self).__init__()
        self.drop = nn.Dropout(p)
        self.linearIn = nn.Linear(insize, nn_hlayer)
        self.linearHidden = nn.Linear(nn_hlayer, nn_hlayer)
        self.linearOut = nn.Linear(nn_hlayer, outsize)
        
    def forward(x):
        x = F.relu(self.drop(self.linearIn(x)))
        x = F.relu(self.drop(self.linearHidden(x)))
        x = F.sigmoid(self.linearOut(x))
        return x



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


# Calculating IQR for loan_amnt and customer_income
loan_iqr = df['loan_amnt'].quantile(0.75) - df['loan_amnt'].quantile(0.25)
loan_lower_limit = df['loan_amnt'].quantile(0.25) - 1.5 * loan_iqr
loan_upper_limit = df['loan_amnt'].quantile(0.75) + 1.5 * loan_iqr


income_iqr = df['customer_income'].quantile(0.75) - df['customer_income'].quantile(0.25)
income_lower_limit = df['customer_income'].quantile(0.25) - 1.5 * income_iqr
income_upper_limit = df['customer_income'].quantile(0.75) + 1.5 * income_iqr

df_filtered = df[(df['loan_amnt'] >= loan_lower_limit) & (df['loan_amnt'] <= loan_upper_limit) &
                 (df['customer_income'] >= income_lower_limit) & (df['customer_income'] <= income_upper_limit)]

# Plotting the data
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_filtered[['loan_amnt', 'customer_income']])
plt.title('Loan Amount and Customer Income')
plt.xlabel('Variables')
plt.ylabel('Amount')
plt.show()

print(f"Loan Amount Limits: {loan_lower_limit:.2f} to {loan_upper_limit:.2f}\nCustomer Income Limits: {income_lower_limit:.2f} to {income_upper_limit:.2f}")

# Catplot for 'loan_amnt' by 'loan_intent'
sns.catplot(data=df_filtered, x='loan_intent', y='loan_amnt', kind='box', height=5, aspect=2)  # Change 'kind' for different plots
plt.title('Loan Amount by Loan Intent')
plt.xticks(rotation=45)
plt.show()

# Catplot for 'loan_amnt' by 'loan_grade'
sns.catplot(data=df_filtered, x='loan_grade', y='loan_amnt', kind='box', height=5, aspect=2)  # Change 'kind' for different plots
plt.title('Loan Amount by Loan Grade')
plt.xticks(rotation=45)
plt.show()

# Catplot for 'customer_income' by 'loan_intent'
sns.catplot(data=df_filtered, x='loan_grade', y='customer_income', kind='box', height=5, aspect=2)  # Change 'kind' for different plots
plt.title('Loan Amount by Loan Grade')
plt.xticks(rotation=45)
plt.show()

# Catplot for 'customer_income' by 'loan_intent'
sns.catplot(data=df_filtered, x='loan_intent', y='customer_income', kind='box', height=5, aspect=2)  # Change 'kind' for different plots
plt.title('Loan Amount by Loan Grade')
plt.xticks(rotation=45)
plt.show()

# numerical_cols = df_filtered.select_dtypes(include=['float64', 'int64']).columns
# plt.figure(figsize=(12, 10))
# sns.pairplot(df_filtered.drop('customer_id', axis=1), diag_kind="kde", hue='loan_intent', corner=True)
# plt.suptitle('Pairplot of Numerical Features', y=1.02)  # Adjusting the title position
# plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(data=df_filtered, x='customer_income', y='loan_amnt', ci=95, marker='+')  # 'ci=None' to remove confidence interval
plt.title('Loan Amount vs Customer Income')
plt.xlabel('Customer Income')
plt.ylabel('Loan Amount')
plt.show()

print('Ending exploratory plots!!!\n')

