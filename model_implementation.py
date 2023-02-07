'''
Data Preprocessing: You need to load the historical data from the CSV files into a data structure such as a Pandas DataFrame. Then, you need to clean the data by removing missing values, dealing with outliers, and transforming the data into a suitable format for the machine learning model.

Split the data into training and testing sets: You can use the train_test_split function from the sklearn library to split the data into training and testing sets. This is important to evaluate the performance of the model.

Build the model: You can use the Keras library to build a recurrent neural network (RNN) model using LSTMs. You need to choose the number of neurons, layers, and activation functions to use in the model.

Compile the model: You need to compile the model using an optimizer, loss function, and evaluation metric.

Train the model: You need to train the model on the training set using the fit function from Keras. You also need to specify the batch size and the number of epochs.

Evaluate the model: You need to evaluate the performance of the model on the testing set using the evaluate function from Keras. You can also generate predictions on the testing set and compare them to the actual prices.

Use the model to make predictions: You can use the predict function from Keras to generate predictions on new data and make decisions in the spot market.
'''

import pandas as pd
import numpy as np
from ta import *
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import ccxt
import threading
import time

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,3"  # specify which GPU(s) to be used

# Load all of the CSV files into individual DataFrames
df_6h = pd.read_csv("6hInfo.csv")
df_5min = pd.read_csv("5minInfo.csv")
df_1m = pd.read_csv("1mInfo.csv")
df_1d = pd.read_csv("1dInfo.csv")
df_12h = pd.read_csv("12hInfo.csv")
df_1h = pd.read_csv("1hInfo.csv")
df_3min = pd.read_csv("3minInfo.csv")
df_4h = pd.read_csv("4hInfo.csv")
df_2h = pd.read_csv("2hInfo.csv")
df_15min = pd.read_csv("15minInfo.csv")
df_1s = pd.read_csv("1sInfo.csv")
df_30min = pd.read_csv("30minInfo.csv")

# Merge all of the DataFrames into a single DataFrame
df = pd.concat([df_6h, df_5min, df_1m, df_1d, df_12h, df_1h, df_3min, df_4h, df_2h, df_15min, df_1s, df_30min])

# Clean the data by removing any missing or irrelevant information
df = df.dropna()

# Preprocess the data by scaling and normalizing the columns that will be used as inputs
scaler = MinMaxScaler()
df[['Open', 'High', 'Low', 'Close', 'Accumulation Distribution', 'Volume', 'Stochastic Oscillator', 'Rate of Change', 'Supertrend', 'relative strength', 'On Balance Volume', 'Macd', 'Ema9', 'ema21', 'ema50', 'ema100', 'ema200', 'moving average', 'upper band', 'lower band']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Accumulation Distribution', 'Volume', 'Stochastic Oscillator', 'Rate of Change', 'Supertrend', 'relative strength', 'On Balance Volume', 'Macd', 'Ema9', 'ema21', 'ema50', 'ema100', 'ema200', 'moving average', 'upper band', 'lower band']])

# Set a random seed for reproducibility
np.random.seed(0)

# Split the data into training and testing sets
msk = np.random.rand(len(df)) < 0.8
train_df = df[msk]
test_df = df[~msk]

# Verify that the training and testing sets have been created
print("Training set shape:", train_df.shape)
print("Testing set shape:", test_df.shape)

#Define the input and output data for the LSTM model
X_train = train_df.drop('Close', axis=1).values
y_train = train_df['Close'].values
X_test = test_df.drop('Close', axis=1).values
y_test = test_df['Close'].values

#Reshape the data into a 3D format to be used by the LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

#Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=1))

#Compile the LSTM model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the LSTM model
model.fit(X_train, y_train, epochs=50, batch_size=32)

#Evaluate the LSTM model on the testing set
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score)


'''Model is created, now we can use real time data to test the LSTM model to make preditions'''
# Collecting historical market data from Binance
'''exchange = ccxt.binance({
    'rateLimit': 2000,
    'enableRateLimit': True,
    'apiKey': '5xMyscOc36LImLRv8BJqzeLDPAJM6sKScUAL1hO6Lo7ykXk92Q1XUNgDcRQDRbU9',
    'secret': 'MNRpusclopFORQTWHHCofw4hbL0UmuNNRMM7TCL5KZeufZ8dAuTgZU9sXsr1AHjV',
    'options': {
        'adjustForTimeDifference': True,
    },
})'''

exchange = ccxt.binance({'enableRateLimit': True})
exchange.set_sandbox_mode(True)

stop_loss = 0.05 # Stop loss at 5%
take_profit = 0.1 # Take profit at 10%

# Implementing position sizing
position_size = 0.01 # Position size of 0.01 BTC

# Implementing risk-reward ratio
risk_reward_ratio = 2 # Will only enter a trade if the potential profit is at least twice the potential loss


# Implementing the trading bot
while True:
    ohlcv5m = exchange.fetch_ohlcv('BTC/USDT', '5m', limit=720)
    ohlcv6h = exchange.fetch_ohlcv('BTC/USDT', '6h', limit=720)
    ohlcv1m = exchange.fetch_ohlcv('BTC/USDT', '1m', limit=720)
    ohlcv1d = exchange.fetch_ohlcv('BTC/USDT', '1d', limit=720)
    ohlcv12h = exchange.fetch_ohlcv('BTC/USDT', '12h', limit=720)
    ohlcv1h = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=720)
    ohlcv3m = exchange.fetch_ohlcv('BTC/USDT', '3m', limit=720)
    ohlcv4h = exchange.fetch_ohlcv('BTC/USDT', '4h', limit=720)
    ohlcv2h = exchange.fetch_ohlcv('BTC/USDT', '2h', limit=720)
    ohlcv15m = exchange.fetch_ohlcv('BTC/USDT', '15m', limit=720)
    ohlcv1s = exchange.fetch_ohlcv('BTC/USDT', '1s', limit=720)
    ohlcv30m = exchange.fetch_ohlcv('BTC/USDT', '30m', limit=720)
        
    ohlcv5m = pd.DataFrame(ohlcv5m)
    ohlcv5m = ohlcv5m.iloc[: , 1:]
    ohlcv6h = pd.DataFrame(ohlcv6h)
    ohlcv6h = ohlcv6h.iloc[: , 1:]
    ohlcv1m = pd.DataFrame(ohlcv1m)
    ohlcv1m = ohlcv1m.iloc[: , 1:]
    ohlcv1d = pd.DataFrame(ohlcv1d)
    ohlcv1d = ohlcv1d.iloc[: , 1:]
    ohlcv12h = pd.DataFrame(ohlcv12h)
    ohlcv12h = ohlcv12h.iloc[: , 1:]
    ohlcv1h = pd.DataFrame(ohlcv1h)
    ohlcv1h = ohlcv1h.iloc[: , 1:]
    ohlcv3m = pd.DataFrame(ohlcv3m)
    ohlcv3m = ohlcv3m.iloc[: , 1:]
    ohlcv4h = pd.DataFrame(ohlcv4h)
    ohlcv4h = ohlcv4h.iloc[: , 1:]
    ohlcv2h = pd.DataFrame(ohlcv2h)
    ohlcv2h = ohlcv2h.iloc[: , 1:]
    ohlcv15m = pd.DataFrame(ohlcv15m)
    ohlcv15m = ohlcv15m.iloc[: , 1:]
    ohlcv1s = pd.DataFrame(ohlcv1s)
    ohlcv1s = ohlcv1s.iloc[: , 1:]
    ohlcv30m = pd.DataFrame(ohlcv30m)
    ohlcv30m = ohlcv30m.iloc[: , 1:]
    
    #print(ohlcv5m)

    ohlcv5m.to_csv("5mcsv.csv", index=True)
    ohlcv6h.to_csv("6hcsv.csv", index=True)
    ohlcv1m.to_csv("1mcsv.csv", index=True)
    ohlcv1d.to_csv("1dcsv.csv", index=True)
    ohlcv12h.to_csv("12hcsv.csv", index=True)
    ohlcv1h.to_csv("1hcsv.csv", index=True)
    ohlcv3m.to_csv("3mcsv.csv", index=True)
    ohlcv4h.to_csv("4hcsv.csv", index=True)
    ohlcv2h.to_csv("2hcsv.csv", index=True)
    ohlcv15m.to_csv("15mcsv.csv", index=True)
    ohlcv1s.to_csv("1scsv.csv", index=True)
    ohlcv30m.to_csv("30mcsv.csv", index=True)

    os.system("python indicators_calc.py 5mcsv.csv 5m.csv")
    os.system("python indicators_calc.py 6hcsv.csv 6h.csv")
    os.system("python indicators_calc.py 1mcsv.csv 1m.csv")
    os.system("python indicators_calc.py 1dcsv.csv 1d.csv")
    os.system("python indicators_calc.py 12hcsv.csv 12h.csv")
    os.system("python indicators_calc.py 1hcsv.csv 1h.csv")
    os.system("python indicators_calc.py 3mcsv.csv 3m.csv")
    os.system("python indicators_calc.py 4hcsv.csv 4h.csv")
    os.system("python indicators_calc.py 2hcsv.csv 2h.csv")
    os.system("python indicators_calc.py 15mcsv.csv 15m.csv")
    os.system("python indicators_calc.py 1scsv.csv 1s.csv")
    os.system("python indicators_calc.py 30mcsv.csv 30m.csv")

    # Load historical market data from csv files
    ohlcv_1d = pd.read_csv("1d.csv")
    ohlcv_1h = pd.read_csv("1h.csv")
    ohlcv_1m = pd.read_csv("1m.csv")
    ohlcv_1s = pd.read_csv("1s.csv")
    ohlcv_2h = pd.read_csv("2h.csv")
    ohlcv_3m = pd.read_csv("3m.csv")
    ohlcv_4h = pd.read_csv("4h.csv")
    ohlcv_5m = pd.read_csv("5m.csv")
    ohlcv_6h = pd.read_csv("6h.csv")
    ohlcv_12h = pd.read_csv("12h.csv")
    ohlcv_15m = pd.read_csv("15m.csv")
    ohlcv_30m = pd.read_csv("30m.csv")

    # Concatenate all the data into one DataFrame
    df = pd.concat([ohlcv_1d, ohlcv_1h, ohlcv_1m, ohlcv_1s, ohlcv_2h, ohlcv_3m, ohlcv_4h, ohlcv_5m, ohlcv_6h, ohlcv_12h, ohlcv_15m, ohlcv_30m], axis=0)
    final_data = df
    print(final_data.shape)
    final_data = final_data.loc[:, ~final_data.columns.str.contains('^Unnamed')]
    
    '''
    order_book = exchange.fetch_order_book('BTC/USDT', limit=720)
    # Adding the order book information to the indicators
    final_data.append(order_book['bids'][0][0])
    final_data.append(order_book['asks'][0][0])
    '''
    
    # Making a prediction
    final_data = scaler.transform(final_data)
    final_data = np.expand_dims(final_data, axis=1)
    prediction = model.predict(final_data)
    
    # Collecting the current market data
    ticker = exchange.fetch_ticker('BTC/USDT')
    current_price = ticker['last']
    print("Last price:", current_price)
    print("prediction: ", prediction[~np.isnan(prediction)])
    np.savetxt("prediction.csv", prediction[~np.isnan(prediction)], delimiter=",")

    
    # Calculating potential profit and loss
    """
    potential_profit = (current_price - prediction) * position_size
    non_nan_values = potential_profit[~np.isnan(potential_profit)]
    np.savetxt("potential_profit.csv", non_nan_values, delimiter=",")
    print("non nan values from potential profit: ", non_nan_values)
    potential_loss = (prediction - current_price) * position_size
    non_nan_values = potential_loss[~np.isnan(potential_loss)]
    np.savetxt("potential_loss.csv", non_nan_values, delimiter=",")
    print("non nan values from potential loss: ", non_nan_values)
    """
    
    # Define a threshold to determine potential profit and potential loss
    threshold = 0.5

    # Convert predictions to binary classifications using the threshold
    binary_predictions = np.where(prediction >= threshold, 1, 0)

    # Calculate potential profit and potential loss
    potential_profit = np.sum(binary_predictions)
    potential_loss = binary_predictions.shape[0] - potential_profit
    potential_loss = (current_price - potential_loss[~np.isnan(potential_loss)]) * position_size
    print(potential_loss)
    potential_profit = -(potential_profit[~np.isnan(potential_profit)] - current_price) * position_size
    print(potential_profit)
    
    '''
    result = np.all(np.isnan(potential_profit))
    if result:
        print("potential_profit has only NaN values")
    else:
        print("potential_profit has some non-NaN values")
        
    result = np.all(np.isnan(potential_loss))
    if result:
        print("potential_loss has only NaN values")
    else:
        print("potential_loss has some non-NaN values")
    '''

    # Checking risk-reward ratio
    if potential_profit/potential_loss >= risk_reward_ratio:
        # Executing the trade
        if prediction+current_price > current_price:
            exchange.create_limit_buy_order('BTC/USDT', position_size, current_price*(1+take_profit), {'stopLoss': current_price*(1-stop_loss)})
            print("Buy order")
            print("Buy price:", current_price*(1+take_profit))
            print("StopLoss:", current_price*(1-stop_loss))
        elif prediction+current_price < current_price:
            exchange.create_limit_sell_order('BTC/USDT', position_size, current_price*(1-take_profit), {'stopLoss': current_price*(1+stop_loss)})
            print("Sell order")
            print("sell price: ", current_price*(1-take_profit))
            print("StopLoss:", current_price*(1-stop_loss))

    # Wait for some time
    time.sleep(60)
