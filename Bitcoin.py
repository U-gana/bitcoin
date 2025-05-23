# importing libries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# load data 
df =  pd.read_csv("coin_Bitcoin.csv")

# data processing 
df.index = pd.to_datetime(df["Date"])
filtered_df = df.drop(['SNo', 'Name', 'Symbol','Date'], axis=1)

from sklearn.ensemble import RandomForestRegressor
all_features = ['High', 'Low', 'Open', 'Close', 'Marketcap', "Volume"]


# assingning feture and target variables
x = filtered_df[all_features].fillna(0)
# Removing the last row since it wont be import for trainin as it predicts for the next day 
x = x.iloc[:-1]

y = filtered_df['Close'].shift(-1)
y = y.iloc[:-1]

# split data to train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2 )

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', LinearRegression())
])

pipeline.fit(x_train, y_train)
pred = pipeline.predict(x_test)
Mae = mean_absolute_error(pred, y_test)
print(Mae)