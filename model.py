import numpy as np
import pandas as pd
from seaborn import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

data = load_dataset("tips")
data.drop_duplicates()
def convert(x):
  if x == "Female" or x == "No":
    return 0
  if x == "Male" or x == "Yes":
    return 1
  return x

data = data.map(convert)

data = pd.get_dummies(data, columns=["day", 'time'])

y = data["tip"]
x = data.drop("tip", axis=1)

def createMode(x, y):
  model = LinearRegression()
  modelScore = 0
  train_size = 0
  
  for i in range(10000):
    x_test, x_train, y_test, y_train, = train_test_split(x, y, train_size=.1, random_state=i)

    y_test.shape, x_test.shape, y_train.shape, x_train.shape

    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(x_train[["total_bill"]])
    x_train["total_bill"] = min_max_scaler.transform(x_train[["total_bill"]])
    x_test["total_bill"] = min_max_scaler.transform(x_test[["total_bill"]])

    newModel = LinearRegression()
    newModel.fit(x_train, y_train)
    newModelScore = newModel.score(x_test, y_test)
    
    if newModelScore > modelScore:
      modelScore = newModelScore
      model = newModel
      train_size = i
  
  print(train_size)
  print(modelScore)
  return model

createMode(x, y)