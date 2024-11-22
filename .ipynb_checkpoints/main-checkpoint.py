import os
import csv

input_dir = os.environ['SM_INPUT_DIR']
input_text = ''

print('###### opening train')
with open(input_dir + "/data/training/house_price_regression_dataset.csv", 'r') as fp:
    # lines = len(fp.readlines())
    # print('Total Number of lines:', lines)
    input_text = str(len(fp.readlines()))

print('###### closed train')


print('###### opening model')
model_dir = os.environ['SM_MODEL_DIR']
with open(model_dir + '/model.txt', 'w') as f:
    f.write('model cio')
print('###### closing model')
    
output_dir = os.environ['SM_MODEL_DIR']
with open(output_dir + '/output.txt', 'w') as f:
    f.write('Ciao output')
with open(output_dir + '/output_csv.txt', 'w') as f:
    f.write(input_text)

##########
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib


file_path = input_dir + "/data/training/house_price_regression_dataset.csv"
data = pd.read_csv(file_path)


X = data.drop(columns='House_Price')
y = data['House_Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, output_dir + '/output.joblib')

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)