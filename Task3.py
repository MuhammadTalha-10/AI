import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_excel('weather.xlsx')

# Remove leading/trailing whitespaces from column names
data.columns = data.columns.str.strip()

# Remove the 'timestamp' column and handle missing values
data = data.dropna()
X = data.drop(['timestamp', 'Temperature'], axis=1)
y = data['Temperature']

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Initialize empty lists to store the data for the animation
timestamps = []
actual_values = []
predicted_values = []

# Define the update function for the animation
def update(frame):
    # Get the current timestamp
    timestamp = data.iloc[frame]['timestamp']
    
    # Get the actual value and predicted value for the current timestamp
    actual_value = y_test.iloc[frame]
    predicted_value = model.predict([X_test[frame]])
    
    # Append the data to the lists
    timestamps.append(timestamp)
    actual_values.append(actual_value)
    predicted_values.append(predicted_value)
    
    # Clear the plot
    ax.clear()
    
    # Plot the actual values and predicted values
    ax.plot(timestamps, actual_values, label='Actual')
    ax.plot(timestamps, predicted_values, label='Predicted')
    
    # Set plot title and labels
    ax.set_title('Temperature Prediction Over Time')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Temperature')
    
    # Add legend
    ax.legend()
    
    # Adjust the layout to prevent overlapping of labels
    plt.tight_layout()

# Create the animation
animation = FuncAnimation(fig, update, frames=len(X_test), interval=200, repeat=False)

# Show the plot
plt.show()
