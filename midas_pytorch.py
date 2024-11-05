# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 19:39:34 2024

@author: JMSA
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Dataset builder class
class Data(Dataset):
    def __init__(self, df, target_col):
        # Dropping client ID
        #df = df.drop(['customer_id', 'historical_default'], axis=1)
        #to_dummies = ['loan_intent', 'loan_grade', 'home_ownership']
        #df_final = pd.get_dummies(data=df, columns=to_dummies)
        #bool_cols = df_final.select_dtypes(include='bool').columns
        #df_final[bool_cols] = df_final[bool_cols].astype(float)
        
        # Prepared to separate the features and targets.
        df_feat = df[['customer_age', 'customer_income', 'employment_duration', 'loan_amnt',
                            'loan_int_rate', 'term_years', 'cred_hist_length', 'loan_intent_DEBTCONSOLIDATION', 
                            'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 
                            'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE',
                            'loan_grade_A', 'loan_grade_B', 'loan_grade_C', 'loan_grade_D',
                            'loan_grade_E', 'home_ownership_MORTGAGE', 'home_ownership_OTHER',
                            'home_ownership_OWN', 'home_ownership_RENT']]
        
        df_target = df[target_col]
        df_target.replace({'NO DEFAULT': 1, 'DEFAULT': 0}, inplace=True)
        
        # Converting to torch tensors.
        self.x = torch.from_numpy(df_feat.to_numpy()).type(torch.FloatTensor)
        self.y = torch.from_numpy(df_target.to_numpy()).type(torch.FloatTensor)
        
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# The neural network class
class classifierNN(nn.Module):
    def __init__(self, insize, outsize, nn_hlayer, p=0.2):
        super(classifierNN, self).__init__()
        self.drop = nn.Dropout(p)
        self.linearIn = nn.Linear(insize, nn_hlayer)
        self.linearHidden = nn.Linear(nn_hlayer, nn_hlayer)
        self.linearHidden2 = nn.Linear(nn_hlayer, nn_hlayer)
        self.linearHidden3 = nn.Linear(nn_hlayer, nn_hlayer)
        self.linearOut = nn.Linear(nn_hlayer, outsize)
        
    def forward(self, x):
        x = F.relu(self.linearIn(x))
        x = F.relu(self.drop(self.linearHidden(x)))
        x = F.relu(self.drop(self.linearHidden2(x)))
        x = F.relu(self.drop(self.linearHidden3(x)))
        x = torch.sigmoid(self.linearOut(x)) 
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

# plt.figure(figsize=(10, 6))
# sns.regplot(data=df_filtered, x='customer_income', y='loan_amnt', ci=95, marker='+')  # 'ci=None' to remove confidence interval
# plt.title('Loan Amount vs Customer Income')
# plt.xlabel('Customer Income')
# plt.ylabel('Loan Amount')
# plt.show()

print('Ending exploratory plots!!!\n')

#Cleaning up data before the rest.
df_filtered = df_filtered.drop(['customer_id', 'historical_default'], axis=1)  # Dropping client_id and historical_default. client_id does not influence the outcome and historical_data is useless it has not enough information.
to_dummies = ['loan_intent', 'loan_grade', 'home_ownership']
df_filtered = pd.get_dummies(data=df_filtered, columns=to_dummies)
bool_cols = df_filtered.select_dtypes(include='bool').columns
df_filtered[bool_cols] = df_filtered[bool_cols].astype(float)

# Extracting the cases that need prediction
df_to_predict = df_filtered[df_filtered['Current_loan_status'].isna()]
df_filtered = df_filtered.dropna(subset=['Current_loan_status'], axis=0).reset_index(drop=True)

# Dropping all rows that contain NaN
df_filtered = df_filtered.dropna().reset_index(drop=True)


# Normalizing values to stabilize the response of the network.

scaler = MinMaxScaler()
res=list(df_filtered.columns)
res.remove('Current_loan_status')
df_normalized = scaler.fit_transform(df_filtered[res].copy())
df_normalized = pd.DataFrame(df_normalized, columns=res)
df_normalized['Current_loan_status'] = df_filtered['Current_loan_status']

# Creating the Dataset instance.
dataset = Data(df_normalized, 'Current_loan_status')

# Splitting the Dataset into a Training Dataset and a Validation Dataset, with a rate of 80% for Training and 20% for Validation.
train_size = int(0.8*len(dataset))
validation_size = int(len(dataset) - train_size)

traindata, validationdata = random_split(dataset, [train_size, validation_size])
trainLoader = DataLoader(dataset=traindata, batch_size=256, shuffle=True)
validLoader = DataLoader(dataset=validationdata, batch_size=512, shuffle=False)

# Creating a model instance
input_size = traindata[0][0].numel()
model = classifierNN(input_size, 1, 1024, p=0.2)

# Creating criterion, and optimizer.
lr = 0.00001
momentum = 0.000005
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr) #, momentum=momentum)

# Performance measuring variables, loss, accuracy and number of epochs.
LOSS = []
LOSSavg = []
LOSSstd = []
accuracy = []
accavg = []
accstd = []
valLOSS = []
valLOSSavg = []
valLOSSstd = []
valAccuracy = []
valAccuracyAvg = []
valAccuracystd = []
epochs = 1000

# Training cycle
start_time = time.time()
for epoch in range(epochs):
    for x, y in trainLoader:
        x, y = x.float(), y.float()
        model.train()
        optimizer.zero_grad()
        yhat = model(x)
        loss = criterion(yhat, y.view(-1, 1))
        loss.backward()
        optimizer.step()
        LOSS.append(loss.item())
        y2 = (yhat >= 0.5).float()
        accuracy.append((y2.view(-1) == y.view(-1)).float().mean())
    accavg.append(np.array(accuracy).mean())
    accstd.append(np.array(accuracy).std())
    accuracy = []
    LOSSavg.append(np.array(LOSS).mean())
    LOSSstd.append(np.array(LOSS).std())
    LOSS = []
    
    with torch.no_grad():
        for x, y in validLoader:
            model.eval()
            yhat = model(x)
            valloss = criterion(yhat, y.view(-1, 1))
            y2 = (yhat >= 0.5).float()
            valAccuracy.append((y2.view(-1) == y.view(-1)).float().mean())
            valLOSS.append(valloss.item())
            
    valLOSSavg.append(np.array(valLOSS).mean())
    valLOSSstd.append(np.array(valLOSS).std())
    valLOSS = []
    valAccuracyAvg.append(np.array(valAccuracy).mean())
    valAccuracystd.append(np.array(valAccuracy).std())
    valAccuracy = []
    
    if epoch%10 == 0:                          # This cycle prints out the epoch number every 10 epochs.
        epoch_time = time.time()
        epoch_time = (epoch_time - start_time)/60             # Converts from seconds to minutes.
        print(f'Passing epoch {epoch:.0f} after {epoch_time:.2f} minutes.')
 
end_time = time.time()   
total_time = (end_time - start_time)/60
print(f"Training time: {total_time:.2f} minutes")

# This Dataframe contains the statistical characteristics of every epoch for training and validation.
trainValStatistics = pd.DataFrame({'Train Accuracy Average':accavg, 'Train Accuracy Std':accstd, 
                                   'Train Loss Average': LOSSavg, 'Train Loss Std': LOSSstd, 
                                   'Validation Accuracy Average': valAccuracyAvg, 'Validation Accuracy std': valAccuracystd, 
                                   'Validation Loss Average': valLOSSavg, 'Validation Loss Std':valLOSSstd})

x = range(len(trainValStatistics))
plt.figure(figsize=(14,10))
sns.lineplot(data=trainValStatistics, x=x, y='Train Accuracy Average', label='Training Accuracy Average')
plt.fill_between(x, trainValStatistics['Train Accuracy Average'] - trainValStatistics['Train Accuracy Std'], trainValStatistics['Train Accuracy Average'] + trainValStatistics['Train Accuracy Std'], alpha=0.3, label='Train Accuracy StD')
sns.lineplot(data=trainValStatistics, x=x, y='Train Loss Average', label='Training Loss Average')
plt.fill_between(x, trainValStatistics['Train Loss Average'] - trainValStatistics['Train Loss Std'], trainValStatistics['Train Loss Average'] + trainValStatistics['Train Loss Std'], alpha=0.3, label='Train Loss StD')
sns.lineplot(data=trainValStatistics, x=x, y='Validation Accuracy Average', label='Validation Accuracy Average')
plt.fill_between(x, trainValStatistics['Validation Accuracy Average'] - trainValStatistics['Validation Accuracy std'], trainValStatistics['Validation Accuracy Average'] + trainValStatistics['Validation Accuracy std'], alpha=0.3, label='Validation Accuracy StD')
sns.lineplot(data=trainValStatistics, x=x, y='Validation Loss Average', label='Validation Loss Average')
plt.fill_between(x, trainValStatistics['Validation Loss Average'] - trainValStatistics['Validation Loss Std'], trainValStatistics['Validation Loss Average'] + trainValStatistics['Validation Loss Std'], alpha=0.3, label='Validation Loss StD')
plt.show()


################################################################ Prediction ###########################################################################

# Predicting if the clients that we took from the Dataset are going to go into Default or not. 

# First we clean and fix a few things, we remove the 'Current_loan_status' because that is our target, and there is a "nan" value that is going to be substituted by the average value.

#######################################################################################################################################################

df_to_predict.loc[22742, 'loan_int_rate'] = df['loan_int_rate'].mean()    # This is a cell that I saw that has a NaN value. This is not the most versatile code. I know.

df_to_predict.drop('Current_loan_status', axis=1)                         # Dropping the target column, it does not have any information anyway.
df_to_predict_normalized=scaler.transform(df_to_predict[res])             # This line normalizes the input values, so that the model only gets values between 0 and 1.

print('The predictions for the 4 cases that had "NaN" in the :')

for ii in range(len(df_to_predict_normalized)):
    print(df_to_predict.iloc[ii])
    print('result: ')
    print('No Default\n\n') if model(torch.tensor(df_to_predict_normalized[3]).type(torch.FloatTensor)).item() == 1.0 else print('Default\n\n')
    
