import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Function to load and preprocess the dataset
def load_data(file_path):
    # Load the data with pandas assuming the first row is the header
    data = pd.read_csv(file_path)
    return data

# Function to read text from file based on selected plant
def read_plant_info(selected_plant):
    plant_info_files = {
        "Bhilai Steel Plant (BSP)": "/Users/akshaaykumar/Downloads/bsp.txt",
        "Durgapur Steel Plant (DSP)": "/Users/akshaaykumar/Desktop/BTP/dsp.txt",
        "Rourkela Steel Plant (RSP)": "/Users/akshaaykumar/Desktop/BTP/rsp.txt",
        "Bokaro Steel Plant (BSL)": "/Users/akshaaykumar/Desktop/BTP/bokaro.txt"

        # Add more plant names and their respective text files here
        # "Plant_Name": "plant_info_file.txt"
    }
    
    if selected_plant in plant_info_files:
        with open(plant_info_files[selected_plant], 'r') as file:
            plant_info = file.read()
        return plant_info
    else:
        return "Information not available for this plant."

def plot_energy_consumption(data, selected_plant):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Year/Plant'], data[selected_plant], marker='o', color='b')
    ax.set_title(f"Energy Consumption for {selected_plant}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Energy Consumption")
    ax.grid(True)
    plt.xticks(rotation='vertical')
    st.pyplot(fig)
    

# Function to forecast energy consumption
def forecast_energy_consumption(data, selected_plant):
    plant_data = data[[selected_plant]].values
    plant_data = plant_data.astype('float32')
    
    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    plant_data = scaler.fit_transform(plant_data)
    
    # Split into train and test sets
    train_size = int(len(plant_data) * 0.8)
    test_size = len(plant_data) - train_size
    train, test = plant_data[0:train_size], plant_data[train_size:len(plant_data)]
    
    # Convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    
    # Reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 1
    X_train, y_train = create_dataset(train, time_step)
    X_test, y_test = create_dataset(test, time_step)
    
    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Invert predictions
    train_predict = scaler.inverse_transform(train_predict)
    y_train = scaler.inverse_transform([y_train])
    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform([y_test])
    
    # Predict the next year
    last_year_data = plant_data[-time_step:].reshape(1, time_step, 1)
    forecast = model.predict(last_year_data)
    forecast = scaler.inverse_transform(forecast)
    
    return forecast[0][0]

# Streamlit App
def main():
    st.title("Energy Consumption Forecasting")
    
    # Load data
    file_path = '/Users/akshaaykumar/Downloads/4.csv'
    data = load_data(file_path)

    # User input for selecting the plant
    st.subheader("Select Plant for Forecasting")
    selected_plant = st.selectbox("Choose a plant", data.columns[1:])
    
    if st.button("Forecast"):
        # Forecast energy consumption
        forecasted_energy = forecast_energy_consumption(data, selected_plant)
        st.write(f"Forecasted Energy Consumption for the next year: {forecasted_energy}")

    plot_energy_consumption(data, selected_plant)

    # Display plant information
    plant_info = read_plant_info(selected_plant)
    st.write("Plant Information:")
    st.write(plant_info)
    
    

if __name__ == "__main__":
    main()
