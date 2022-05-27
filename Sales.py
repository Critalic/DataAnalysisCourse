import pandas as pand
import seaborn as sea
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression


dataFrame = pand.read_csv("C:\\Users\\START\\Downloads\\kc_house_data.csv")

# x_train, x_test, y_train, y_test = train_test_split(dataFrame[['sqft_living']], dataFrame['price'], test_size=0.25)

lr = LinearRegression()
f = dataFrame[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]
lr.fit(f, dataFrame['price'])
# cross_score = cross_val_score(lr, x_test, y_test, cv=4)
print(lr.score(f, dataFrame['price']))
print(f.dtypes)