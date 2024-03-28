import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


file_path = os.path.join(os.path.dirname(__file__), 'sample-data/Titanic-Dataset.csv')
df = pd.read_csv(file_path)
columns = df.columns.tolist()

# Deleting Missing Values
df.dropna(inplace=True)

# Data Cleaning
df.drop_duplicates(inplace=True)

for col in columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.lower()

# Data splitting
y = df['Survived']
X = df.drop(columns=['Survived'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
