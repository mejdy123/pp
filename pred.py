import pandas as pd
from sklearn.model_selection import train_test_split

# Chargement
iris_df = pd.read_csv("../iris.csv")

# Séparation train - test
y = iris_df['class']
X = iris_df.drop(labels='class', axis=1)
train_X_iris, test_X_iris, train_y_iris, test_y_iris =  train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)