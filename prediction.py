import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.models import load_model
import joblib

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
    except Exception as e:
        print(f"Error in duration conversion: {e}")
        return None

test_df = pd.read_csv('C:\\Users\\sidd\\Downloads\\TEST.csv')

test_df['duration'] = test_df['duration'].apply(iso8601_duration_to_seconds)
category = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8}

test_df['category'] = test_df['category'].map(category)  # Use the same mapping as in training data

test_df = test_df[test_df['views'] != 'F']
test_df = test_df[test_df['likes'] != 'F']
test_df = test_df[test_df['dislikes'] != 'F']
test_df = test_df[test_df['comment'] != 'F']

test_df[['views', 'likes', 'comment', 'dislikes']] = test_df[['views', 'likes', 'comment', 'dislikes']].apply(pd.to_numeric)
test_df['published'] = LabelEncoder().fit_transform(test_df['published'])
test_df['vidid'] = LabelEncoder().fit_transform(test_df['vidid'])
loaded_rf_model = joblib.load('rf_model.pkl')
loaded_ann_model = load_model('my_model.keras')

X_test = test_df.drop(['vidid'], axis=1)

scaler = MinMaxScaler()
X_test = scaler.fit_transform(X_test)  

rf_predictions = loaded_rf_model.predict(X_test)

ann_predictions = loaded_ann_model.predict(X_test).flatten()

test_df['adview_rf'] = rf_predictions
test_df['adview_ann'] = ann_predictions


test_df.to_csv('path_to_save_predictions.csv', index=False)


