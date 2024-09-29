# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 19:39:34 2024

@author: JMSA
"""

import numpy as np
import pandas as pd





df = pd.read_csv('LoansDatasest.csv')

# Converting relevant columns from object to numeric
df['customer_income']=df['customer_income'].str.replace(',','').str.replace('£','')
df['customer_income'] = pd.to_numeric(df['customer_income'], errors='coerce')

df['loan_amnt']=df['loan_amnt'].str.replace(',','').str.replace('£','')
df['loan_amnt'] = pd.to_numeric(df['loan_amnt'], errors='coerce')
