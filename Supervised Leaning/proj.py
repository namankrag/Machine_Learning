import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"Prime AIML\Supervised ML\loan_approval_data.csv")

# print(df.head())
# print(df.describe())
# print(df.info())
# print(df.isnull().sum())

#           Handling missing values 

categorical_colm = df.select_dtypes(include=["object"]).columns
numerical_colm = df.select_dtypes(include=["number"]).columns

from sklearn.impute import SimpleImputer

num_imp = SimpleImputer(strategy="mean")
df[numerical_colm] = num_imp.fit_transform(df[numerical_colm])

cat_imp = SimpleImputer(strategy="most_frequent")
df[categorical_colm] = cat_imp.fit_transform(df[categorical_colm])

# print(df.isnull().sum())

#           EDA

#pie chart
class_cnt = df["Loan_Approved"].value_counts()
plt.pie(class_cnt, labels=["No", "Yes"], autopct="%1.1f%%")
plt.title("Is loan approved or not ?")

#box plot

# fig, axes = plt.subplots(2, 2)
# sns.boxenplot(ax = axes[0, 0], data=df, x="Loan_Approved", y="Applicant_Income")
# sns.boxenplot(ax = axes[0, 1], data=df, x="Loan_Approved", y="Credit_Score")
# sns.boxenplot(ax = axes[1, 0], data=df, x="Loan_Approved", y="DTI_Ratio")
# sns.boxenplot(ax = axes[1, 1], data=df, x="Loan_Approved", y="Savings")
# plt.tight_layout()

fig, axes = plt.subplots(2, 2)
sns.boxplot(ax = axes[0, 0], data=df, x="Loan_Approved", y="Applicant_Income")
sns.boxplot(ax = axes[0, 1], data=df, x="Loan_Approved", y="Credit_Score")
sns.boxplot(ax = axes[1, 0], data=df, x="Loan_Approved", y="DTI_Ratio")
sns.boxplot(ax = axes[1, 1], data=df, x="Loan_Approved", y="Savings")
plt.tight_layout()

plt.show()

# Remove Applicant_ID
df = df.drop("Applicant_ID", axis=1)