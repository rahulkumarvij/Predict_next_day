import numpy as np
from os import environ
environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import pandas as pd
nifty_data = pd.read_csv("NIFTY 50_Data.csv")
nifty_data['Date'] =  pd.to_datetime(nifty_data['Date'], format='%d %b %Y')
nifty_data = nifty_data.dropna()
nifty_data = nifty_data.sort_values('Date')
nifty_data.head()
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
nifty_data = nifty_data.drop("Date",axis=1)
nifty_data
# Scale the data
scaler = MinMaxScaler()

# Define the number of days in one row
window = 20
print("1")
# Create a new DataFrame to store the transformed data
transformed_data = pd.DataFrame()

# Iterate over the rows of the original DataFrame
for i in range(len(nifty_data) - window):
    # Create a new row with the data from the last 20 days
    row = nifty_data.iloc[i:i+window]
    # Add the new row to the transformed DataFrame
    transformed_data = transformed_data.append(pd.DataFrame(row.values.flatten().tolist()).transpose())

# Reset the index of the transformed DataFrame
transformed_data.reset_index(drop=True, inplace=True)
transformed_data

transformed_data = scaler.fit_transform(transformed_data)

# Split the data into training and test sets
train_data = transformed_data[:len(transformed_data)-10]
test_data = transformed_data[len(transformed_data)-10:]
type(train_data)

X_train = train_data[:,:76]
Y_train = train_data[:,76:]
X_test = test_data[:,:76]
Y_test = test_data[:,76:]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_train.shape

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

model = Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=4))
model.compile(optimizer='adam',loss='mean_squared_error')
print("2")
model.fit(X_train,Y_train,epochs=10,batch_size=32)

predicted_stock_price = model.predict(X_test)

a = scaler.inverse_transform(np.concatenate((X_test,predicted_stock_price), axis=None).reshape(-1,80))[:,76:]
a = a[:][:][:,3:]
a

b = scaler.inverse_transform(test_data)[:,76:]
b = b[:][:][:,3:]
b

import matplotlib.pyplot as plt

plt.plot(b.flatten(), color = 'black', label = ' Stock Price')
plt.plot(a.flatten(), color = 'green', label = 'Predicted Stock Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
