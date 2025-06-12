import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib

df = pd.read_csv('tweets.csv')

df['word_count'] = df['content'].apply(lambda x: len(str(x).split()))
df['char_count'] = df['content'].apply(lambda x: len(str(x)))
df['sentiment'] = df['content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df['has_media'] = df['media'].notnull().astype(int)
df['hour'] = pd.to_datetime(df['datetime']).dt.hour
df['day_of_week'] = pd.to_datetime(df['datetime']).dt.dayofweek

le = LabelEncoder()
df['company_encoded'] = le.fit_transform(df['inferred company'])

features = ['word_count', 'char_count', 'sentiment', 'has_media', 'hour', 'day_of_week', 'company_encoded']
X = df[features]
y = df['likes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print("RMSE:", rmse)

joblib.dump(model, 'like_predictor.pkl')
joblib.dump(le, 'label_encoder.pkl')
