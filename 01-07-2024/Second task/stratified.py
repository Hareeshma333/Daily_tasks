import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import KBinsDiscretizer

# Load our data
df = pd.read_csv('housing.csv')

# Split the dataset into features and target
x = df[['size', 'bedrooms']].values
y = df['price'].values

# Discretize the continuous target variable into bins
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
y_binned = discretizer.fit_transform(y.reshape(-1, 1)).astype(int).ravel()

# Define the model
model = LinearRegression()

# Define Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5)
mae_scores = []

for train_index, test_index in skf.split(x, y_binned):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model
    model.fit(x_train, y_train)
    
    # Predict the test set
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mae_scores.append(mae)
    
average_mae = np.mean(mae_scores)
print(f"Average Mean Absolute Error: {average_mae}")

