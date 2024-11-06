# Midas_Pytorch

This is an exercise I did to practice the application of Machine Learning and Deep Learning to a possible real scenario. In this case, we want to see if a client who requests a loan can pay it back based on a series of parameters.

Used Libs: **time, numpy, matplotlib, seaborn, pandas, pytorch, and sklearn**

# About Dataset
## Loan Default Prediction Dataset

This dataset contains information about customer loans, including customer demographics, loan details, and default status. The dataset can be used for various data analysis and machine learning tasks, such as predicting loan default risk. The dataset consists of the following columns:

* customer_id: Unique identifier for each customer 
* customer_age: Age of the customer
* customer_income: Annual income of the customer
* home_ownership: Home ownership status (e.g., RENT, OWN, MORTGAGE)
* employment_duration: Duration of employment in months
* loan_intent: Purpose of the loan (e.g., PERSONAL, EDUCATION, MEDICAL, VENTURE)
* loan_grade: Grade assigned to the loan
* loan_amnt: Loan amount requested
* loan_int_rate: The interest rate of the loan
* term_years: Loan term in years
* historical_default: Indicates if the customer has a history of default (Y/N)
* cred_hist_length: Length of the customer's credit history in years
* Current_loan_status: Current status of the loan (DEFAULT, NO DEFAULT) <------------- This is our target!!!! :)

__Dataset source: [Loan-Dataset](https://www.kaggle.com/datasets/prakashraushan/loan-dataset)__

__Dataset by: [Raushan](https://www.kaggle.com/prakashraushan)__ 
