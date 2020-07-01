import os
import csv
import sys
import time
import keras
import plotly
import calendar
import numpy as np
import pandas as pd
import urllib.request
import tensorflow as tf
from keras import optimizers
from keras.models import Model
from sklearn import preprocessing
import plotly.graph_objects as go
import alpaca_trade_api as tradeapi
from keras.callbacks import EarlyStopping
from datetime import date, datetime, timedelta
from alpha_vantage.timeseries import TimeSeries
from keras.layers import Dense, Dropout, LSTM, Input, Activation
# Above libaries are not in any order other than length

np.random.seed(4)
tf.random.set_seed(4)


# Save csv data file
# Get data from alpha_vantage via the api key
def save_dataset(symbol):
    # Get current stock symbol
    stock_symbol = symbol
    api_key = 'YOUR_API_KEY'
    # https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=HD&apikey=7WS55POSQ00X17DW&datatype=csv
    # https://query1.finance.yahoo.com/v7/finance/download/HD?period1=1559079357&period2=1590701757&interval=1d&events=history
    
    print('Beginning file download...')
    url_link = ("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=" + str(stock_symbol) + "&apikey=" + str(api_key) + "&datatype=csv")
    #url_link = ("https://query1.finance.yahoo.com/v7/finance/download/" + str(symbol) + "?period1=1559079357&period2=1590701757&interval=1d&events=history")
    save_file_name = ("C:\\Path\\To\\Downloaded\\Stock_Files\\" + str(stock_symbol) + "_stock.csv")
    urllib.request.urlretrieve(url_link, save_file_name)
    print('Download finshed.')
    
    
# Check downloaded data and update the history dataset if needed    
def add_to_history(symbol):
    # Start the data updating
    print("Adding new stock data to history")

    input_file = ('C:\\Path\\To\\Downloaded\\Stock_Files\\' + str(symbol) + '_stock.csv')
    output_file = ('C:\\Path\\To\\Downloaded\\Stock_Files\\' + str(symbol) + '_History.csv')

    opened_new_csv = pd.read_csv(input_file)
    New_Data = opened_new_csv
    New_Data = New_Data.values
    
    opened_old_csv = pd.read_csv(output_file)
    Old_Data = opened_old_csv
    Old_Data = Old_Data.values
    
    Count = 0 
    for line in New_Data[:,0]:
        if line not in Old_Data:
            Count += 1
    print("Days worth of new data to be added: " + str(Count))

    New_Input = []
    i = 0
    
    while i < Count:
        New_Input.append(New_Data[i][:])
        i += 1
    
    New_Input = New_Input[::-1]

    Write_File = open(output_file, 'a')
    
    for line in New_Input:
        Length = len(line)
        j = 0
        while j < Length:
            Write_File.write(str(line[j]))
            if j < (Length - 1):
                Write_File.write(',')
            j += 1
        Write_File.write('\n')
    
    Write_File.close()
    print("New stock data added successfully.")


# Number of days to look back on and predict off of
Look_Back = 15
# How many days into the future the network should predict prices for
Prediction_Days = 10


# Hold the normal multiplier for open(0), high(1), low(2), close(3), and volume(4) (in that order) in normal multiplier
def Normalize_Data(data_array):
    # Store original shape of data array
    shape = data_array.shape
    
    # Array of zeros the same size as the data array
    # We will fill it with the normalized data
    Normalized_Array = np.zeros(shape)
    
    # Get number of columns in data array
    Columns = int(data_array.shape[1])

    i = 0
    k = 0
    # Create a normal multiplier value for each column of data
    # We need these multipliers when we de-normalize the data for actual prices
    # Normal multiplier for open(0), high(1), low(2), close(3), and volume(4) in that order
    while (i < Columns):
        Normal = np.linalg.norm(data_array[:,i])
        Normal_Multiplier.append(Normal)
        i += 1

    # For each data number, normalize it, then feed it into the new data array
    while (k < Columns):
        Normalized_Array[:,k] = data_array[:,k]/Normal_Multiplier[k]
        k += 1
    
    return Normalized_Array


# Reverse the order of an array
def reverse_array(data_array):
    Reversed_Array = data_array[::-1]

    return Reversed_Array


# Convert the csv file to a dataset for the model
def csv_to_dataset(test_set_name):
    # Folder of all stock csv files
    main_directory = ("C:\\Path\\To\\Downloaded\\Stock_Files\\")

    for csv_file_path in list(os.listdir(main_directory)):
        if csv_file_path == (str(Current_Stock) + "_History.csv"):
            
            # Full directory to stock file
            directory_full = (str(main_directory) + str(csv_file_path))
            
            # Open csv stock data file
            opened_csv = pd.read_csv(directory_full)
            
            # Drop all other columns other than the timestamp column
            # The model has no need for the date, but it will be helpful for graphing
            Time_Stamp_Data = opened_csv.drop(["open", "high", "low", "close", "volume"], axis=1)
            Time_Stamp_Data = Time_Stamp_Data.values
            Time_Stamp_Array.append(Time_Stamp_Data)

            # Drop timestamp column so that the network only intakes actual price and volume data
            Input_Data = opened_csv.drop("timestamp", axis=1)          
            Input_Data = Input_Data.values
            
            # Reverse the order of the data so that the oldest date comes first
            reverse_array(Input_Data)
            reverse_array(Time_Stamp_Data)
            
    # Normalize all the data
    Normalized_Input_Data.append(Normalize_Data(Input_Data))
    
    return Normalized_Input_Data, Time_Stamp_Data


# Convert normal array back into real numbers
def normalized_to_plain(data_array):
    # Get shape of data array
    Shape = data_array.shape
    
    # Array of zeros the same size as the data array
    # We will fill it with the normalized data
    Real_Number_Array = np.zeros(Shape)
    
    # Get number of columns in data array
    Columns = int(data_array.shape[1])
    
    j = 0
    while (j < Columns):
        Real_Number_Array[:,j] = (data_array[:,j] * Normal_Multiplier[j])
        j += 1
        
    return Real_Number_Array


# Predict the next prices of the current stock
def predict_prices(num_prediction):
    # Get original shape of array
    Shape = Validation_Input_Data.shape
    
    # Get only past number of values from dataset
    # The past number is defined by Look_Back
    Look_Back_Data = Validation_Input_Data.reshape(Validation_Input_Data.shape[0], Validation_Input_Data.shape[1])
    Look_Back_Data = Look_Back_Data[-Look_Back:,:]
    Guess = Look_Back_Data[-Look_Back:,:]
    
    Last_Price = Look_Back_Data[-1,3]
    
    # Get original shape of array
    Shape = Look_Back_Data.shape

    # Get the number of columns
    Columns = int(Look_Back_Data.shape[1])
    
    # Separate each column into its own array for easy manipulation
    Column_1_Data = Look_Back_Data[:,0]
    Column_2_Data = Look_Back_Data[:,1]
    Column_3_Data = Look_Back_Data[:,2]
    Column_4_Data = Look_Back_Data[:,3]
    Column_5_Data = Look_Back_Data[:,4]
    
    # Creat a list of each column array
    Column_List = [Column_1_Data, Column_2_Data, Column_3_Data, Column_4_Data, Column_5_Data]
    
    Count = 0
    while (Count < num_prediction):
        i = 0
        while (i < Columns):
            Column_Data = Column_List[i]
            
            j = 0
            Sum = 0
            Average = 0
            while (j < Look_Back):
                Sum += Column_Data[-j]
                j += 1
            # 5 averages for each column
            Average = (Sum/Look_Back)
            
            # Append each average to its respective column
            if (i == 0):
                Column_1_Data = np.append(Column_1_Data, Average)
                
            if (i == 1):
                Column_2_Data = np.append(Column_2_Data, Average)
                
            if (i == 2):
                Column_3_Data = np.append(Column_3_Data, Average)
                
            if (i == 3):
                Column_4_Data = np.append(Column_4_Data, Average)
                
            if (i == 4):
                Column_5_Data = np.append(Column_5_Data, Average)
                
            Column_List = [Column_1_Data, Column_2_Data, Column_3_Data, Column_4_Data, Column_5_Data]
            
            i += 1
        Count += 1
    # Data is now just in 5 separate columns
    # Create array with correct 2d shape
    Look_Back_Input = np.empty(((Look_Back + num_prediction), 5))
    
    # Combine column data back into a single 2d array
    Item_Index = 0
    while (Item_Index < (Look_Back + num_prediction)):
        Column_Index = 0
        for item in Column_List:
            # Prints each item in the row (each column)
            # Item_Index is the row
            # Column_Index is the column
            Look_Back_Input[Item_Index, Column_Index] = item[Item_Index]
            Column_Index += 1
        Item_Index += 1  
    
    # Convert data array back into 3d for the model input
    Predicted_Input = np.reshape(Look_Back_Input, Look_Back_Input.shape + (1,))
    Guess_Input  = np.reshape(Guess, Guess.shape + (1,))
    
    # Predict prices over the entire Look_Back dataset
    Prediction_Output = model.predict(Guess_Input)
    # Get the last day, then all the predicted future days prices
    Prediction_Output = Prediction_Output[-num_prediction:]
    Prediction_Output = np.insert(Prediction_Output, 0, Last_Price)

    return Prediction_Output
    

# Predict the next dates starting from today
def predict_dates(num_prediction):
    # Get last date in dataset
    # Predict days starting with current date (this won't let it start with a weekend)
    Todays_Date = date.today()
    
    # Initialize empty list for dates
    Prediction_Dates = []
    
    Period = num_prediction
    prediction_dates = pd.date_range(Todays_Date, periods=Period)
    i = 0
    while i < Period:
        Current_Date = str(prediction_dates.date[i])
        Prediction_Dates.append(Current_Date)
        i += 1

    return Prediction_Dates


def graph_data(time_array, data_array, time_array2, data_array2, time_array3, data_array3, data_array4, data_array5):

    Layout_1 = go.Layout(
        title = (str(Current_Stock) + " Stock Closing Prices"),
        xaxis = {'title' : "Date"},
        yaxis = {'title' : "Closing Price (USD)"}
    )

    # x is the time array, y is the actual data array (closing values in this case)
    # Graph of the real closing prices
    Line_1 = go.Scatter(
        x = time_array,
        y = data_array,
        mode = 'lines',
        name = 'Real Prices',
        line_color = 'rgb(0,0,0)'
    )
    
    # Graph of the predicted prices during training/validation
    Line_2 = go.Scatter(
        x = time_array2,
        y = data_array2,
        mode = 'lines',
        name = 'Validation Prices',
        line_color = 'rgb(0,128,0)'
    )
    
    # Predicted future prices
    Line_3 = go.Scatter(
        x = time_array3,
        y = data_array3,
        mode = 'lines',
        name = 'Predicted Future Prices',
        line_color = 'rgb(128,0,128)'
    )
    
    # Short term moving average
    Line_4 = go.Scatter(
        x = time_array,
        y = data_array4,
        mode = 'lines',
        name = 'Short Term Moving Average',
        line_color = 'rgb(255,0,0)'
    )
    
    # Long term moving average
    Line_5 = go.Scatter(
        x = time_array,
        y = data_array5,
        mode = 'lines',
        name = 'Long Term Moving Average',
        line_color = 'rgb(0,0,255)'
    )
    
    fig = go.Figure(data=[Line_1, Line_2, Line_3, Line_4, Line_5], layout=Layout_1)
    fig.show()
    return
    

# Calculate the moving average for a stock    
def moving_average(period, data_array):
    Number_Of_Days = period
    
    Shape = data_array.shape
    Output_Averages = np.zeros(Shape)
   
    k = Number_Of_Days
    while k < (len(data_array) - 1):
        
        # Starting and stopping points for each period
        Start = (k - Number_Of_Days)
        Stop = k
        
        # Get data for that period
        Period_Data = (data_array[Start:Stop])
        
        # Keep up with the position for each average
        Average_Date = k + 1
        # Sum of the prices for that period
        Price_Sum = 0
        i = 0
        
        while i < Number_Of_Days:
            # Add all prices together
            Price_Sum += (Period_Data[i] * Normal_Multiplier[3])
            i += 1
        
        # Calculate the moving average of that period
        Moving_Average = (Price_Sum/Number_Of_Days)
        # Put moving average back into data array for graphing
        Output_Averages[Average_Date] = Moving_Average
        k += 1
        
    return(Output_Averages)
    
    
if __name__ == "__main__":
    
    # Run network for each stock symbol
	# The stock symbols are abitrary. You can use whatever you like
    Stock_List = ['NVDA', 'HD', 'STX', 'GE']
    for item in Stock_List:
        Current_Stock = item
        print("Running Network on stock symbol " + str(item) + ".")
    
        # These need to be initialized inside the loop
        # They need to be empty for each stock symbol
        # Initialize normalized arrays to be used in later functions
        Time_Stamp_Array = []
        Normalized_Input_Data = []
        Normal_Multiplier = []
        Normalized_Array = []
        Real_Number_Array = []
    
        # Download latest data
        save_dataset(Current_Stock)
        
        # Get latest stock data
        add_to_history(Current_Stock)
        
        # Get history for current stock
        Stock_History = (str(Current_Stock) + "_History.csv")
        
        # Convert stock history into dataset to fed to the network
        csv_to_dataset(Stock_History)
        
        # Remove datatypes
        Normalized_Input_Data = Normalized_Input_Data[0]
        
        # Get closing prices from entire dataset
        Normalized_Closing_Price = Normalized_Input_Data[:,3]
        
        # Split the dataset up into training and validation datasets
        # Use the first 75% of data to train on (This is the oldest data)
        # Use the last 25% of data to validate on (This is the most current data)
        Length_Of_Data = len(Normalized_Input_Data)
        Validation_Length = int(Length_Of_Data * 0.25)
        History_Length = (Length_Of_Data - Validation_Length)
        
        # 956 days worth of data for network to train on (as of 6/20/2020)
        History_Data = Normalized_Input_Data[:History_Length]
        
        # 318 days worth of data (most current dates) to be validated on (as of 6/20/2020)
        Validation_Data = Normalized_Input_Data[History_Length:]

        Time_Stamp_Array = Time_Stamp_Array[0]
        # Flatten the time stamp array so that it is one dimension
        Time_Stamp_Array = Time_Stamp_Array.ravel()
        
        # Create neural network architecture
        # Concluded that the first dropout layer(s) need to be more harsh than the latter ones
        Model_Input = Input(shape=(5, 1), name='Model_Input')
        x = LSTM(64, name='Model_0')(Model_Input)
        x = Dropout(0.5, name='Model_dropout_0')(x)
        x = Dense(64, name='Dense_0')(x)
        x = Dropout(0.1, name='Model_dropout_1')(x)
        x = Activation('sigmoid', name='Sigmoid_0')(x)
        x = Dense(1, name='Dense_1')(x)
        Model_Output = Activation('linear', name='Linear_output')(x)
        model = Model(inputs=Model_Input, outputs=Model_Output)
        adam = optimizers.Adam(lr=0.0005)
        model.compile(optimizer=adam, loss='mse')
        
        # Make history and validation datasets 3 dimensional
        History_Input_Data = np.reshape(History_Data, History_Data.shape + (1,))
        Validation_Input_Data = np.reshape(Validation_Data, Validation_Data.shape + (1,))

        # Get just the closing price from the train and validation datasets
        History_Closing_Price = History_Input_Data[:,3]
        Validation_Closing_Price = Validation_Input_Data[:,3]
        
        # Implement an early stopping to the epochs to try and prevent overfitting
        # Once model longer improves its learning, stop it so it doesn't cheat
        Early_Stopping = EarlyStopping(monitor='loss', mode='min', patience=130, verbose=1)
        
        # Train neural network
        # x is equal to the entire dataset after being normalized
        # Entire dataset has shape (number of days worth of data, 5 (columns), 1)
        # y is equal to the closing prices of the dataset
        history = model.fit(x=History_Input_Data, y=History_Closing_Price, batch_size=150, epochs=1150, shuffle=False, validation_data=(Validation_Input_Data, Validation_Closing_Price), verbose=0, callbacks=[Early_Stopping])
        
        # Evaluate how well the network did
        Evaluation = model.evaluate(Validation_Input_Data, Validation_Closing_Price)
        print("The evaluation of the neural network was: " + str(Evaluation) + " (Mean squared error)")

        # Get loss and val_loss data from training and validation respectively
        # Get the percentage of epochs that were not an overfit.
        # val_loss < loss means that it didn't overfit
        Loss = history.history['loss']
        Validation_Loss = history.history['val_loss']
        i = 0
        Overfit = 0
        Good_Fit = 0
        while i < len(Loss):
            if (float(Loss[i])) < (float(Validation_Loss[i])):
                Overfit += 1
            elif (float(Loss[i])) > (float(Validation_Loss[i])):
                Good_Fit += 1
            i += 1
        Good_Fit_Percent = ((Good_Fit / len(Loss)) * 100)
        print("Percentage of epochs that were a goodfit: " + str(Good_Fit_Percent) + "%")
        
        # Use the model to predict the validation data
        # Convert its predictions back into real numbers after
        Predicted_Closing_Price = model.predict(Validation_Input_Data)
        Predicted_Number_Data = Predicted_Closing_Price * Normal_Multiplier[3]
        
        # Get real closing prices for validation data
        Real_Number_Data = normalized_to_plain(Validation_Input_Data)
        Real_Closing_Price = Real_Number_Data[:,3]
        
        # Get dates the reflect the validation data
        Validation_Time = Time_Stamp_Array[History_Length:]
        
        # Predict future dates (really is more of getting them from the calendar)
        Forecast_Dates = predict_dates(Prediction_Days)
        
        # Predict future prices and convert back to real numbers
        Forecast_Prices_Normalized = predict_prices(Prediction_Days)
        Forecast_Prices = Forecast_Prices_Normalized * Normal_Multiplier[3]
        
        # Flatten out the array so it is one dimension
        Flat_Closing_Price = Real_Closing_Price.ravel()
        Flat_Predicted_Price = Predicted_Number_Data.ravel()
        Forecast_Prices = Forecast_Prices.ravel()
        
        # Time period for the short moving average
        Short_Period = 20
        # Time period for the long moving average
        Long_Period = 100
       
        # When the short term crosses above the long term its a buy signal (golden cross)
        # Short term below the long term sell signal (dead cross)
        Short_Moving = moving_average(Short_Period, Normalized_Closing_Price)
        # Get  moving average for validation dataset
        Short_Moving = Short_Moving[History_Length:]
        # Flatten array to one dimension for graphing
        Short_Moving = Short_Moving.ravel()

        # Get long moving average for validation dataset
        Long_Moving = moving_average(Long_Period, Normalized_Closing_Price)
        # Get  moving average for validation dataset
        Long_Moving = Long_Moving[History_Length:]
        # Flatten array to one dimension for graphing
        Long_Moving = Long_Moving.ravel()

        # Create a graph of the actual closing prices everyday plus the future predicted prices and moving averages
        # This graph starts at the date of the validation data
        # Training data isn't used since it would just show the real prices
        graph_data(Validation_Time, Flat_Closing_Price, Validation_Time, Flat_Predicted_Price, Forecast_Dates, Forecast_Prices, Short_Moving, Long_Moving)
        
        # Golden Cross or Dead cross methods
        if Short_Moving[-1] < Long_Moving[-1]:
            print("Short average is lower than the long. Indicates a sell.")
            # Dead cross
        elif Short_Moving[-1] > Long_Moving[-1]:
            print("Short average is higher than the long. Indicates a buy.")
            # Golden cross

