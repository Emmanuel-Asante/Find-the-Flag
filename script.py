# Import modules
import codecademylib3_seaborn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Load data into a dataframe
flags = pd.read_csv("flags.csv", header = 0)

# Print out flags column names
print(flags.columns)

# Print out the first five rows of the dataset
print(flags.head())

# Create a label
labels = flags[["Landmass"]]

# Select specific columns
data = flags[["Red", "Green", "Blue", "Gold", "White", "Black", "Orange", "Circles", "Crosses", "Saltires", "Quarters", "Sunstars", "Crescent", "Triangle"]]

# Split dataframe
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)

# Create an empty list
scores = []

# create a for loop
for i in range(1, 21):
  # Create a model object
  tree = DecisionTreeClassifier(random_state = 1, max_depth = i)
  # Train the model
  tree.fit(train_data, train_labels)
  # Print out the accuracy of the model
  scores.append(tree.score(test_data, test_labels))

# Create a line plot
plt.plot(range(1, 21), scores)

# Show plot
plt.show()