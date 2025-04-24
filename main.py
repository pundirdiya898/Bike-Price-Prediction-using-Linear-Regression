# IMPORT LIBRARY
import pandas as pd
import numpy as np

# Import CSV file as Dataframe
df = pd.read_csv(r'https://github.com/YBI-Foundation/Dataset/raw/main/Bike%20Prices.csv')

# Get the first five rows of dataframe
df.head()

# Get Information of Dataframe
df.info()

# Get Missing values drop
df = df.dropna()

# Get Summary Statistics
df.describe()

# Get Categories and counts of Categorical Variables
df[['Brand']].value_counts()
df[['Model']].value_counts()
df[['Seller_Type']].value_counts()
df[['Owner']].value_counts()

# Get Column Names
df.columns
df.shape

# Get Encoding of Categorical features
df.replace({'Seller_Type':{'Individual':0,'Dealer':1}},inplace=True)
df.replace({'Owner':{'1st owner':0,'2nd owner':1,'3rd owner':2,'4th owner':3}},inplace=True)
ax = pd.get_dummies(X,columns=['Seller_Type','Owner'],drop_first=True)

# Define target variable (y) and features (X)
y = df['Selling_Price']
y.shape
y
X = df[[ 'Year', 'Seller_Type', 'Owner','KM_Driven', 'Ex_Showroom_Price']]
X

# Get Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =  train_test_split(X,y,test_size=0.3,random_state=2529)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

# Get Model Train
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

# Get Model Prediction
y_pred = model.predict(X_test)
y_pred.shape
y_pred

# Get Model Evaluation
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
mean_squared_error(y_test,y_pred)
mean_absolute_error(y_test,y_pred)
r2_score(y_test,y_pred)

# Get Visualization of Actual VS Predicted Results
import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price VS Predicted Price")
plt.show()

# Get Future Predictions
df_new = df.sample(1)
df_new
df_new.shape
X_new = df_new.drop(['Brand','Model','Selling_Price'],axis=1)
y_pred_new = model.predict(X_new)
y_pred_new
