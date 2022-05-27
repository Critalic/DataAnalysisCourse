import pandas as pand
import numpy as np

headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
           "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
           "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower",
           "peak-rpm", "city-mpg", "highway-mpg", "price"]
dataFrame = pand.read_csv("C:\\Users\\START\\Downloads\\module_5_auto.csv", )
missingData = dataFrame.isnull()
dataFrame.replace("?", np.nan, inplace=True)

dataFrame["stroke"].replace(np.nan, dataFrame["stroke"].mean(axis=0), inplace=True)
dataFrame["num-of-doors"].replace(np.nan, dataFrame['num-of-doors'].value_counts().idxmax(), inplace=True)
dataFrame.dropna(subset=['price'], axis=0, inplace=True)
dataFrame['highway-L/100km'] = 235 / dataFrame['highway-mpg']
# dataFrame.drop(['highway-L/100km'], axis=1, inplace=True)
dataFrame['height-normalized'] = dataFrame['height'] / dataFrame['height'].max()
dummy_var = pand.get_dummies(dataFrame['aspiration'])

dummy_var.rename(columns={'std':'naturally-aspirated'}, inplace=True)
dataFrame = pand.concat([dataFrame, dummy_var], axis=1)
dataFrame.drop('aspiration', axis=1, inplace=True)
print(dataFrame.to_string())