import pandas as pd
import numpy as np
import random

numerical_data = pd.read_csv("Numerical_Data.csv")
text_data = pd.read_csv("Column_Cleaned_Data.csv")

numerical_data['Latitude'] = text_data['Latitude']
numerical_data['Longitude'] = text_data['Longitude']
numerical_data['Speed Limit'] = text_data['Speed Limit']
numerical_data['Vehicle Year'] = text_data['Vehicle Year']
numerical_data['datetime time'] = pd.to_datetime(text_data['Time'])
numerical_data['datetime date'] = pd.to_datetime(text_data['Date'])
numerical_data['day of the week'] = numerical_data['datetime date'].dt.dayofweek
numerical_data['day of the year'] = numerical_data['datetime date'].dt.dayofyear
numerical_data['week of the year'] = numerical_data['datetime date'].dt.weekofyear
numerical_data['month'] = numerical_data['datetime date'].dt.month
numerical_data['year'] = numerical_data['datetime date'].dt.year
numerical_data['hour'] = numerical_data['datetime time'].dt.hour
numerical_data['minute'] = numerical_data['datetime time'].dt.minute
numerical_data['time of day in minutes'] = numerical_data['minute'] + numerical_data['hour'] * 60
numerical_data['accident binary'] = 1

del numerical_data['datetime time']
del numerical_data['datetime date']
del numerical_data['Date']
del numerical_data['Time']
del numerical_data['hour']
del numerical_data['minute']
del numerical_data['Location']

print(text_data.dtypes)

numerical_data.to_csv("normalized_numerical_data.csv", index=False)
