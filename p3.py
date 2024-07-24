import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error # type: ignore

# Load the dataset
file_path = './AAPL.csv'
data = pd.read_csv(file_path)

# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Plot the closing price over time
plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Close'], label='Closing Price')
plt.title('AAPL Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data['Close'][:train_size], data['Close'][train_size:]

# Fit the ARIMA model with different orders
order = (5, 1, 0)  # Example order, this can be tuned
model = ARIMA(train, order=order)
model_fit = model.fit()

# Make predictions
predictions = model_fit.forecast(steps=len(test))
test_index = data['Date'][train_size:]

# Calculate evaluation metrics
mse = mean_squared_error(test, predictions)
mae = mean_absolute_error(test, predictions)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Close'], label='Actual Closing Price')
plt.plot(test_index, predictions, label='Predicted Closing Price', color='red')
plt.title('AAPL Stock Price Prediction using ARIMA')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
