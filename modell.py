import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn import metrics
import re
def iso8601_duration_to_seconds(duration_str):
    try:
        pattern = re.compile(r'PT(?:(\d+)M)?(?:(\d+)S)?')
        match = pattern.match(duration_str)
        if match:
            minutes = match.group(1)
            seconds = match.group(2)
            minutes = int(minutes) if minutes else 0
            seconds = int(seconds) if seconds else 0
            return minutes * 60 + seconds
        return None
    except:
        return None

df = pd.read_csv('C:\\Users\\sidd\\Downloads\\Train2.csv')
df['duration'] = df['duration'].apply(iso8601_duration_to_seconds)
category = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8}
df['category'] = df['category'].map(category)

df = df[df['views'] != 'F']
df = df[df['likes'] != 'F']
df = df[df['dislikes'] != 'F']
df = df[df['comment'] != 'F']

df['adview'] = pd.to_numeric(df['adview'])
df['views'] = pd.to_numeric(df['views'])
df['likes'] = pd.to_numeric(df['likes'])
df['comment'] = pd.to_numeric(df['comment'])
df['dislikes'] = pd.to_numeric(df['dislikes'])

df['published'] = LabelEncoder().fit_transform(df['published'])
df['vidid'] = LabelEncoder().fit_transform(df['vidid'])


y_train = pd.DataFrame(data=df.iloc[:, 1].values, columns=['target'])
df = df.drop(['adview', 'vidid'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(df, y_train, test_size=0.2, random_state=42)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


def print_error(x_test, y_test, model):
    if hasattr(model, 'predict'):
        predictions = model.predict(x_test).flatten()
        mae = metrics.mean_absolute_error(y_test, predictions)
        mse = metrics.mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        print(f"{model.__class__.__name__}:")
        print("Mean Absolute Error:", mae)
        print("Mean Squared Error:", mse)
        print("Root Mean Squared Error:", rmse)
    else:
        print(f"{model.__class__.__name__} does not have a predict method.")


models = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=200, max_depth=25, min_samples_split=15, min_samples_leaf=2),
    "SVR": SVR(),
}

for name, model in models.items():
    model.fit(x_train, y_train)
    print_error(x_test, y_test, model)

ann = Sequential([
    Dense(6, activation="relu", input_shape=(x_train.shape[1],)),
    Dense(6, activation="relu"),
    Dense(1)
])
ann.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
history = ann.fit(x_train, y_train, epochs=100, verbose=1)
ann.summary()
print_error(x_test, y_test, ann)
ann.save('my_model.keras') 

import joblib  
joblib.dump(models["RandomForestRegressor"], 'rf_model.pkl')







