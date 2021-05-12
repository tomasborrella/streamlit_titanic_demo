# Imports

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Functions


def manipulate_df(df):
	# Update sex column to numerical
	df['Sex'] = df['Sex'].map(lambda x: 0 if x == 'male' else 1)
	# Fill the nan values in the age column
	df['Age'].fillna(value=df['Age'].mean(), inplace=True)
	# Create a first class column
	df['FirstClass'] = df['Pclass'].map(lambda x: 1 if x == 1 else 0)
	# Create a second class column
	df['SecondClass'] = df['Pclass'].map(lambda x: 1 if x == 2 else 0)
	# Create a second class column
	df['ThirdClass'] = df['Pclass'].map(lambda x: 1 if x == 3 else 0)
	# Select the desired features
	df = df[['Sex', 'Age', 'FirstClass', 'SecondClass', 'ThirdClass', 'Survived']]
	return df


# Main

print("Loading data")
train_df = pd.read_csv("data/titanic_train.csv")

print("Transforming data")
train_df = manipulate_df(train_df)
features = train_df[['Sex', 'Age', 'FirstClass', 'SecondClass', 'ThirdClass']]
survival = train_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(features, survival, test_size=0.3, random_state=42)

print("Scaling data")
scaler = StandardScaler()
train_features = scaler.fit_transform(X_train)
test_features = scaler.transform(X_test)

print("Training model")
model = LogisticRegression()
model.fit(train_features, y_train)

print("Calculating model results")
train_score = model.score(train_features, y_train)
test_score = model.score(test_features, y_test)
y_predict = model.predict(test_features)

print(f"\tTrain Set Score: {round(train_score, 3)}")
print(f"\tTest Set Score: {round(test_score,3)}")

print(f"Saving scaler to disk")
filename = 'trained_models/titanic_scaler_logistic_regression.pkl'
pickle.dump(scaler, open(filename, 'wb'))

print(f"Saving model to disk")
filename = 'trained_models/titanic_model_logistic_regression.pkl'
pickle.dump(model, open(filename, 'wb'))
