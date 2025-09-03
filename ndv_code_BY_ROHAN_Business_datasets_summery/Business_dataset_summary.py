# Assignment 5: Hypothesis Testing on Business Data

# 1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, ttest_ind, chi2_contingency, f_oneway

# 2. Load Dataset
df = pd.read_csv("/sales.csv")  # Replace with actual file name
df.head()

# 3. Summary Statistics
print("Descriptive Statistics:\n", df.describe())
print("\nMedian Values:\n", df.median(numeric_only=True))
print("\nMode Values:\n", df.mode(numeric_only=True).iloc[0])

# 4. Hypothesis 1: One-Sample t-test
# H₀: Average Monthly Sales = 50000
# H₁: Average Monthly Sales ≠ 50000

t_stat, p_val = ttest_1samp(df['Monthly_Sales'], 50000)
print("\nOne-sample t-test:")
print("T-statistic:", t_stat)
print("P-value:", p_val)
if p_val < 0.05:
    print("Reject the null hypothesis.")
else:
    print("Fail to reject the null hypothesis.")

# 5. Hypothesis 2: Two-Sample t-test (e.g., salary comparison by gender)
# H₀: Mean salary is equal between genders
# H₁: Mean salary is different between genders

male_salary = df[df['Gender'] == 'Male']['Salary']
female_salary = df[df['Gender'] == 'Female']['Salary']
t_stat2, p_val2 = ttest_ind(male_salary, female_salary)
print("\nTwo-sample t-test:")
print("T-statistic:", t_stat2)
print("P-value:", p_val2)
if p_val2 < 0.05:
    print("Reject the null hypothesis.")
else:
    print("Fail to reject the null hypothesis.")

# 6. Hypothesis 3: Chi-Square Test (Gender vs Attrition)
# H₀: Gender and Attrition are independent
# H₁: Gender and Attrition are dependent

contingency_table = pd.crosstab(df['Gender'], df['Attrition'])
chi2, p_val3, dof, expected = chi2_contingency(contingency_table)
print("\nChi-square test:")
print("Chi2:", chi2)
print("P-value:", p_val3)
if p_val3 < 0.05:
    print("Reject the null hypothesis.")
else:
    print("Fail to reject the null hypothesis.")

# 7. ANOVA: Compare Salary across Departments
# H₀: Mean salary is the same across departments
# H₁: At least one department has a different mean salary

groups = [group['Salary'].dropna() for name, group in df.groupby('Department')]
f_stat, p_val4 = f_oneway(*groups)
print("\nANOVA test:")
print("F-statistic:", f_stat)
print("P-value:", p_val4)
if p_val4 < 0.05:
    print("Reject the null hypothesis.")
else:
    print("Fail to reject the null hypothesis.")

# 8. Visualizations

# Boxplot for Salary across Departments
plt.figure(figsize=(8, 5))
sns.boxplot(x='Department', y='Salary', data=df)
plt.title('Salary Distribution by Department')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Histogram of Monthly Sales
plt.figure(figsize=(7, 5))
sns.histplot(df['Monthly_Sales'], bins=20, kde=True)
plt.title('Distribution of Monthly Sales')
plt.xlabel('Monthly Sales')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 9. Correlation Heatmap
plt.figure(figsize=(8, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# 10. Pivot Table (Bonus)
pivot = df.pivot_table(values='Salary', index='Department', columns='Gender', aggfunc='mean')
print("\nPivot Table - Average Salary by Department and Gender:\n", pivot)
