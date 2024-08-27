from pprint import pprint
import requests
import os
from dotenv import load_dotenv
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from copy import deepcopy


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import InputLayer


# Ensure matplotlib uses a non-interactive backend
plt.switch_backend('Agg')


# Load environment variables from .env file
load_dotenv()


def string_to_obj(date: str) -> datetime.datetime:
   """Convert a date string in 'YYYY-MM-DD' format to a datetime object."""
   split = date.split('-')
   year, month, day = int(split[0]), int(split[1]), int(split[2])
   return datetime.datetime(year=year, month=month, day=day)


def df_to_windowed_df(dataframe: pd.DataFrame, first_date: datetime.datetime, last_date: datetime.datetime, n: int=3) -> pd.DataFrame:
   """Convert dataframe to a windowed dataframe with rolling windows of size n."""
   target_date = first_date
   dates = []
   X, Y = [], []
   last_time = False
  
   while not last_time:
       # Get the subset of the dataframe for the current window
       df_subset = dataframe.loc[:target_date].tail(n+1)
      
       # Check if the window size is valid
       if len(df_subset) != n+1:
           print(f'Error: Window of size {n} is too large for date {target_date}')
           return pd.DataFrame()
      
       values = df_subset['Close'].to_numpy()
       x, y = values[:-1], values[-1]
       dates.append(target_date)
       X.append(x)
       Y.append(y)


       # Determine the next date
       next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
       next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
       next_date_str = next_datetime_str.split('T')[0]
       year, month, day = next_date_str.split('-')
       next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
      
       # Break if we have reached the last date
       if target_date == last_date:
           last_time = True
      
       target_date = next_date
  
   # Create a new dataframe with the windowed data
   ret_df = pd.DataFrame({'Target Date': dates})
   X = np.array(X)
   for i in range(n):
       ret_df[f'Target-{n-i}'] = X[:, i]
   ret_df['Target'] = Y
   return ret_df


def windowed_df_to_date_X_y(windowed_dataframe: pd.DataFrame):
   """Convert a windowed dataframe to date, X, and y arrays."""
   df_as_np = windowed_dataframe.to_numpy()
   dates = df_as_np[:, 0]
   middle_matrix = df_as_np[:, 1:-1]
   X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
   Y = df_as_np[:, -1]
   return dates, X.astype(np.float32), Y.astype(np.float32)


def prepare_future_data(dataframe: pd.DataFrame, last_date: datetime.datetime, n: int=3, steps: int=1500):
   """Prepare data for future predictions."""
   future_dates = []
   X_future = []


   current_date = last_date
   for _ in range(steps):
       df_subset = dataframe.loc[:current_date].tail(n)
       if len(df_subset) != n:
           print(f'Error: Window of size {n} is too large for date {current_date}')
           return None, None
      
       values = df_subset['Close'].to_numpy()
       future_dates.append(current_date + datetime.timedelta(days=1))
       X_future.append(values)
       current_date += datetime.timedelta(days=1)
  
   X_future = np.array(X_future).reshape((len(X_future), n, 1))
   return future_dates, X_future.astype(np.float32)


def lstmPredict(ticker: str) -> str:
   """Train an LSTM model to predict stock prices and generate various plots."""
   ticker = ticker
   url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={os.getenv('API_KEY')}"
   response = requests.get(url)


   if response.status_code == 200:
       data = response.json()
       if 'Time Series (Daily)' in data:
           time_series = data['Time Series (Daily)']
           df = pd.DataFrame.from_dict(time_series, orient='index')
           df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
           df.index = df.index.to_series().apply(string_to_obj)
           df.sort_index(inplace=True)
           df['Close'] = df['Close'].astype(float)
           df.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace=True)


           # Check if there's enough data for the window size
           if len(df) < 2 * (3 + 1):
               print(f"Error: Not enough data points for the window size of {3}")
               return


           first_date = df.index[3]
           last_date = df.index[-4]


           # Plot closing prices
           plt.figure(figsize=(10, 5))
           plt.plot(df.index, df['Close'], label='Closing Price')
           plt.xlabel('Date')
           plt.ylabel('Closing Price')
           plt.title(f'{ticker} Closing Prices')
           plt.legend()
           plt.grid(True)
           plt.savefig(os.path.join('static', 'stock_plot.png'))
           plt.close()


           windowed_df = df_to_windowed_df(df, first_date, last_date, n=3)
           dates, X, y = windowed_df_to_date_X_y(windowed_df)
           q_80 = int(len(dates) * .8)
           q_90 = int(len(dates) * .9)


           # Split data into training, validation, and test sets
           dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
           dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
           dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]


           # Plot train, validation, and test sets
           plt.figure(figsize=(10, 5))
           plt.plot(dates_train, y_train, 'b', label='Train')
           plt.plot(dates_val, y_val, 'g', label='Validation')
           plt.plot(dates_test, y_test, 'r', label='Test')
           plt.legend()
           plt.grid(True)
           plt.title(f'{ticker} Closing Prices')
           plt.savefig(os.path.join('static', 'model_plot.png'))
           plt.close()


           # Build and compile the LSTM model
           model = Sequential([
               InputLayer((3, 1)),
               LSTM(64),
               Dense(32, activation='relu'),
               Dense(32, activation='relu'),
               Dense(1)
           ])
           model.compile(loss='mse', optimizer='adam', metrics=['mean_absolute_error'])
          
           # Train the model
           model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)     


           # Generate training predictions plot
           train_predictions = model.predict(X_train).flatten()
           plt.figure(figsize=(10, 5))
           plt.plot(dates_train, train_predictions, label='Training Predictions')
           plt.plot(dates_train, y_train, label='Training Observations')
           plt.legend()
           plt.grid(True)
           plt.xlabel('Date')
           plt.ylabel('Closing Price')
           plt.title(f'{ticker} Training Predictions')
           plt.savefig(os.path.join('static', 'training_predictions.png'))
           plt.close()


           # Generate validation predictions plot
           val_predictions = model.predict(X_val).flatten()
           plt.figure(figsize=(10, 5))
           plt.plot(dates_val, val_predictions, label='Validation Predictions')
           plt.plot(dates_val, y_val, label='Validation Observations')
           plt.legend()
           plt.grid(True)
           plt.xlabel('Date')
           plt.ylabel('Closing Price')
           plt.title(f'{ticker} Validation Predictions')
           plt.savefig(os.path.join('static', 'validation_predictions.png'))
           plt.close()


           # Generate test predictions plot
           test_predictions = model.predict(X_test).flatten()
           plt.figure(figsize=(10, 5))
           plt.plot(dates_test, test_predictions, label='Testing Predictions')
           plt.plot(dates_test, y_test, label='Testing Observations')
           plt.legend()
           plt.grid(True)
           plt.xlabel('Date')
           plt.ylabel('Closing Price')
           plt.title(f'{ticker} Testing Predictions')
           plt.savefig(os.path.join('static', 'testing_predictions.png'))
           plt.close()


           # Prepare future data and generate future predictions plot
           future_steps = 1500
           future_dates, X_future = prepare_future_data(df, last_date, n=3, steps=future_steps)
           if X_future is not None:
               future_predictions = model.predict(X_future).flatten()


               plt.figure(figsize=(10, 5))
               plt.plot(dates_train, train_predictions, label='Training Predictions')
               plt.plot(dates_train, y_train, label='Training Observations')
               plt.plot(dates_val, val_predictions, label='Validation Predictions')
               plt.plot(dates_val, y_val, label='Validation Observations')
               plt.plot(dates_test, test_predictions, label='Testing Predictions')
               plt.plot(dates_test, y_test, label='Testing Observations')
               plt.plot(future_dates, future_predictions, label='Future Predictions')
               plt.legend()
               plt.grid(True)
               plt.xlabel('Date')
               plt.ylabel('Closing Price')
               plt.title(f'{ticker} Overall Predictions')
               plt.savefig(os.path.join('static', 'overall_predictions.png'))
               plt.close()


           return ticker
       else:
           print("Error: 'Time Series (Daily)' not found in response data")
   else:
       print(f"Error: Failed to retrieve data, status code {response.status_code}")


   return None
