 # Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('Salary_Data.csv')  # Make sure this file is in the same folder
print("First 5 records:\n", df.head())
print("\nSummary statistics:\n", df.describe())

# Visualize data: Scatter plot
sns.scatterplot(x='YearsExperience', y='Salary', data=df)
plt.title('Salary vs Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.grid(True)
plt.show()

# Prepare data
X = df[['YearsExperience']]  # Independent variable (2D array)
y = df['Salary']             # Dependent variable

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Plot regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Regression Line Fit')
plt.legend()
plt.grid(True)
plt.show()

# Bonus: Error bars
plt.errorbar(X_test.values.flatten(), y_pred, yerr=np.abs(y_pred - y_test), fmt='o', label='Prediction ± Error')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Prediction with Error Bars')
plt.legend()
plt.grid(True)
plt.show()

# Bonus: Actual vs Predicted Bar Chart
comparison_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
comparison_df.plot(kind='bar', figsize=(8, 6))
plt.title('Actual vs Predicted Salary')
plt.xlabel('Index')
plt.ylabel('Salary')
plt.grid(True)
plt.tight_layout()
plt.show()

# Bonus: User input for prediction
try:
    experience = float(input("Enter years of experience: "))
    predicted_salary = model.predict([[experience]])
    print(f"Predicted Salary for {experience} years of experience is ₹{predicted_salary[0]:,.2f}")
except:
    print("Invalid input.")
