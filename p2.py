import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import requests
from io import StringIO

url = 'https://www.dropbox.com/scl/fi/9golfaeyr4se8taejttw3/AAPL.csv?rlkey=kx388jwf81f6g0has9cghufr1&st=c5cu0hr2&dl=0'
response = requests.get(url)
data = pd.read_csv(StringIO(response.text))

# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Plot the closing price over time
plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Close'], label='Closing Price')
plt.title('Stock Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Plot the trading volume over time
plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Volume'], label='Trading Volume', color='orange')
plt.title('Stock Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.grid(True)
plt.show()

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data['Close'][:train_size], data['Close'][train_size:]

# Fit the ARIMA model
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Make predictions
predictions = model_fit.forecast(steps=len(test))
test_index = data['Date'][train_size:]

# Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE)
mse = mean_squared_error(test, predictions)
mae = mean_absolute_error(test, predictions)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Close'], label='Actual Closing Price')
plt.plot(test_index, predictions, label='Predicted Closing Price', color='red')
plt.title('Stock Price Prediction using ARIMA')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate code complexity
def calculate_complexity():
    """
    Calculate the time complexity of fitting the ARIMA model.
    This is a rough estimation as the exact complexity can vary.
    """
    n = len(train)
    # ARIMA time complexity is generally O(n^2) for fitting
    time_complexity = n ** 2
    
    return time_complexity

# Calculate the complexity of the model fitting process
complexity = calculate_complexity()
print(f'Estimated Time Complexity of ARIMA model fitting: O({complexity})')

# Adding comments and documentation
"""
The script performs the following steps:

1. Load the dataset and convert the 'Date' column to datetime format.
2. Plot the closing price and trading volume over time to observe trends.
3. Split the data into training (80%) and testing (20%) sets.
4. Fit an ARIMA(5, 1, 0) model to the training data and make predictions on the testing data.
5. Calculate the Mean Squared Error (MSE) and Mean Absolute Error (MAE) to evaluate the model's performance.
6. Plot the actual vs. predicted closing prices.
7. Conclude the effectiveness of the ARIMA model.
8. Calculate the rough time complexity of the ARIMA model fitting process.
"""
