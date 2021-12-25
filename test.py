import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Plotting and Visualizing data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
class regLinear:
    def liearReg():
        data = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv', usecols=["Global_Sales", "JP_Sales", "EU_Sales"])
        print(data.describe())

        x = data.iloc[:, 0:1].values
        y = data.iloc[:, 1]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
        print(x[:10])
        print('\n')
        print(y[:10])

        # Model Import and Build
        regressor = LinearRegression()
        regressor.fit(x_train, y_train)

        pred = regressor.predict(x_test)

        # Visualization
        ## Check the fitting on training set
        plt.scatter(x_train, y_train)
        plt.plot(x_train, regressor.predict(x_train), color='black')
        plt.title('Fit on training set')
        plt.xlabel('X-Train')
        plt.ylabel('Y-Train')
        plt.show()


regresja=regLinear.liearReg()
