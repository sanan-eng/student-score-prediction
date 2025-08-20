import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

from cF_1vLRM import y_train

student_dataset_path=pd.read_csv('StudentPerformanceFactors.csv')
print(student_dataset_path)
print(student_dataset_path.isnull().sum())
print("Total Features :",student_dataset_path.columns)
print("Feature Extraction :")
X=student_dataset_path[['Hours_Studied']]
y=student_dataset_path['Exam_Score']
print("Total Entries in Hour_Studies :",student_dataset_path['Hours_Studied'].value_counts().sum())
print("Total Entries in Exam_Score :",student_dataset_path['Exam_Score'].value_counts().sum())
plt.scatter(X,y,color='blue')
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")
plt.show()
print("X shape :",X.shape)
print("y shape :",y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=42)
print("Training set size :",X_train.shape[0])
print("Testing set size :",X_test.shape[0])
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)
print("Model training completed")
from sklearn.metrics import mean_squared_error,r2_score
y_pred_linear=model.predict(X_test)
mse_linear=mean_squared_error(y_test,y_pred_linear)
r2=r2_score(y_test,y_pred_linear)
#linear regression performance
print("Mean Square Error(mse) :",mse_linear)
print("r2_Score (Accuracy) :",r2)
fig,axes=plt.subplots(1,2,figsize=(12,5))
#training data visualization
axes[0].scatter(X_train,y_train,color='blue',label='Actual(Train)')
axes[0].plot(X_train, model.predict(X_train), color='red', linewidth=2, label="Predicted Line")
axes[0].set_title("Linear Regression (Training Data)")
axes[0].set_xlabel("Hours Studied")
axes[0].set_ylabel("Exam Score")
axes[0].legend()
# Testing data visualization
axes[1].scatter(X_test, y_test, color='green', label="Actual (Test)")
axes[1].plot(X_test, y_pred_linear, color='red', linewidth=2, label="Predicted Line")
axes[1].set_title("Linear Regression (Testing Data)")
axes[1].set_xlabel("Hours Studied")
axes[1].set_ylabel("Exam Score")
axes[1].legend()

plt.tight_layout()
plt.show()
#polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.transform(X_test)
poly_model=LinearRegression()
poly_model.fit(X_train_poly,y_train)
y_pred_poly=poly_model.predict(X_test_poly)
mse_poly=mean_squared_error(y_test,y_pred_poly)
r2_poly=r2_score(y_test,y_pred_poly)
print("Polynomial Regression (degree 2) MSE :",mse_poly)
print("Polynomial Regression (degree 2) R2 Score :",r2_poly)
#performance comparison
print("PERFORMANCE COMPARISON:")
#Linear regression
print("Mean Square Error(mse) :",mse_linear)
print("r2_Score (Accuracy) :",r2)
#Polynomial regression
print("Polynomial Regression (degree 2) MSE :",mse_poly)
print("Polynomial Regression (degree 2) R2 Score :",r2_poly)

#With multiple features
print("With Multiple Features")
X_new=student_dataset_path[['Hours_Studied','Attendance','Sleep_Hours']]
y_new=student_dataset_path['Exam_Score']
X_train_new,X_test_new,y_train_new,y_test_new=train_test_split(X_new,y_new,test_size=0.2,random_state=42)
model_new=LinearRegression()
model_new.fit(X_train_new,y_train_new)
y_pred_new=model_new.predict(X_test_new)
mse_new=mean_squared_error(y_test_new,y_pred_new)
r2_new=r2_score(y_test_new,y_pred_new)
print("Linear Regression with Multiple Features MSE :",mse_new)
print("Linear Regression with Multiple Features R2 Score :",r2_new)
