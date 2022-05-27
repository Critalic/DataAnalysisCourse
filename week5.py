from sklearn.linear_model import LinearRegression, Ridge
import pandas as pand
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    ax1 = sea.distplot(RedFunction, hist=False, color="r", label=RedName)
    sea.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()


def PollyPlot(xtrain, xtest, y_train, y_test, lr, poly_transform):
    # poly_transform:  polynomial transformation object

    xmax = max([xtrain.values.max(), xtest.values.max()])

    xmin = min([xtrain.values.min(), xtest.values.min()])

    x = np.arange(xmin, xmax, 0.1)

    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()

lr = LinearRegression()
dataFrame = pand.read_csv("C:\\Users\\START\\Downloads\\module_5_auto.csv")
y_data = dataFrame['price']
x_data = dataFrame.drop('price', 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=0)
# print("number of test samples :", x_test.shape[0])
# print("number of training samples:",x_train.shape[0])

lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)
# print(lr.score(x_test[['horsepower']], y_test))
# print(cross_val_score(lr, x_data[['horsepower']], y_data, cv=4))

Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
# DistributionPlot(y_train, lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]), "Actual Values (Train)", "Predicted Values (Train)", Title)
# DistributionPlot(y_test, lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]), "Actual Values (Test)", "Predicted Values (Test)", Title)

pr1 = PolynomialFeatures(degree=2)
x_train_pr1 = pr1.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
x_test_pr1 = pr1.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
poly1 = LinearRegression().fit(x_train_pr1, y_train)
y_aht = poly1.predict(x_test_pr1)
# DistributionPlot(y_test, y_aht, "sd", "fd", "pakop")


# from tqdm import tqdm
#
# Rsqu_test = []
# Rsqu_train = []
# dummy1 = []
# Alpha = 10 * np.array(range(0, 1000))
# pbar = tqdm(Alpha)
#
# for alpha in pbar:
#     RigeModel = Ridge(alpha=alpha)
#     RigeModel.fit(x_train_pr1, y_train)
#     test_score, train_score = RigeModel.score(x_test_pr1, y_test), RigeModel.score(x_train_pr1, y_train)
#
#     pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})
#
#     Rsqu_test.append(test_score)
#     Rsqu_train.append(train_score)
#
# plt.plot(Alpha,Rsqu_test, label='validation data  ')
# plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
# plt.show()


params = [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000000]}]
RR = Ridge()
grid = GridSearchCV(RR, params, cv=4)
grid.fit(x_train_pr1, y_train)
print(grid.best_estimator_.score(x_test_pr1, y_test))

DistributionPlot(y_test, grid.best_estimator_.predict(x_test_pr1), "a", "b", 'c' )
