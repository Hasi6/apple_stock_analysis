import matplotlib.pyplot as plt

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

# Plot the trading volume over time
plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Volume'], label='Trading Volume', color='orange')
plt.title('AAPL Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.grid(True)
plt.show()
