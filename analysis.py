import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your dataset
# Replace 'your_file.csv' with your actual file name
df = pd.read_csv('OpitmalTimingofBuyingFlightTicket/dataset/Clean_Dataset.csv')

# 2. Ensure dates are in the correct format
# df['booking_date'] = pd.to_datetime(df['booking_date'])
# df['flight_date'] = pd.to_datetime(df['flight_date'])

# 3. Calculate "Days in Advance"
df['days_in_advance'] = df['days_left']

# 4. Group by days to find the average price trend
trend = df.groupby('days_in_advance')['price'].mean()

# 5. Plot the results
plt.figure(figsize=(10, 6))
plt.plot(trend.index, trend.values, color='red', label='Average Price')
plt.title('Average Price vs. Days Before Flight')
plt.xlabel('Days Before Flight')
plt.ylabel('Average Price')
plt.grid(True)
plt.legend()
plt.show()

# 6. Find the cheapest day on average
best_day = trend.idxmin()
best_price = trend.min()
print(f"Best time to buy is {best_day} days before the flight (Price: ${best_price:.2f})")