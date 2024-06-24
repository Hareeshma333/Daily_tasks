#1.Load the dataset from a CSV file named sample_dataset.csv into a Pandas DataFrame. Display the first few rows of the dataset.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

#load the dataset loan_data.csv
df= pd.read_csv('Sample_dataset.csv')

# 2.Generate summary statistics for this dataset. What are the mean and standard deviation of the Sepal Length?
summary_stats = df.describe()


sepal_length_mean = summary_stats.loc['mean', 'Sepal Length (cm)']
sepal_length_std = summary_stats.loc['std', 'Sepal Length (cm)']

summary_stats, sepal_length_mean, sepal_length_std

#3.Check for any missing values in the dataset. How would you handle them if there were any?

missing_values = df.isnull().sum()

missing_values

# 4.Convert the species labels to numerical values using a mapping dictionary. For example, map 'FlowerA' to 0, 'FlowerB' to 1, and 'FlowerC' to 2.

species_mapping = {'FlowerA': 0, 'FlowerB': 1, 'FlowerC': 2}


df['Species'] = df['Species'].map(species_mapping)

df.head()

# 5 Split the dataset into training and testing sets with 70% training data and 30% testing data. Ensure that the split is stratified based on the species.



# Features and target variable
X = df.drop(columns=['Species'])
y = df['Species']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


#6.Train a decision tree classifier on the training data. What parameters would you use for the decision tree?


# Train the decision tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

#7.visualize the trained decision tree.


# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(dt_classifier, feature_names=X.columns, class_names=['FlowerA', 'FlowerB', 'FlowerC'], filled=True)
plt.show()


#  8.Predict the species for the testing data and compute the accuracy.

# Make predictions on the testing data
y_pred = dt_classifier.predict(X_test)

# Compute the accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracy

#9.Generate a classification report and a confusion matrix for the predictions.
# Classification report
class_report = classification_report(y_test, y_pred, target_names=['FlowerA', 'FlowerB', 'FlowerC'])

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

class_report, conf_matrix
