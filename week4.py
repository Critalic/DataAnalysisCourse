from sklearn.linear_model import LinearRegression
import pandas as pand
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

dataFrame = pand.read_csv("C:\\Users\\START\\Downloads\\module_5_auto.csv")

lm1 = LinearRegression()
x = dataFrame[['engine-size']]
y = dataFrame[['price']]
reg = lm1.fit(x, y)
# print(lm1.coef_, lm1.intercept_)

lm2 = LinearRegression()
lm2.fit(dataFrame[['normalized-losses', 'highway-mpg']], dataFrame['price'])
# print(lm2.coef_, lm2.intercept_)

# print(dataFrame[["highway-mpg", 'peak-rpm', 'price']].corr())
# sea.regplot(x='highway-mpg', y = 'price', data=dataFrame)
# plt.ylim(0,)
# plt.show()

# sea.residplot(dataFrame[['highway-mpg']], dataFrame['price'])
# plt.show()

# y_hat = lm2.predict(dataFrame[['normalized-losses', 'highway-mpg']])
# ax1 = sea.distplot(dataFrame['price'], hist=False, color='r', label='Actual value')
# sea.distplot(y_hat, hist=False, color='b', label='Fitted values', ax=ax1)
# plt.title('Actual vs Fitted Values for Price')
# plt.xlabel('Price (in dollars)')
# plt.ylabel('Proportion of Cars')
# plt.show()

x = dataFrame['highway-mpg']
y = dataFrame['price']
f= np.polyfit(x, y, 11)
# print(np.poly1d(f))
# PlotPolly(np.poly1d(f), x, y, 'mpg')

Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
Z = dataFrame[['normalized-losses', 'highway-mpg']].astype(float)
pipe = Pipeline(Input)
pipe.fit(Z, y)
print(pipe.predict(Z)[0:4])

