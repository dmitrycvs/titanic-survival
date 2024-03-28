import os
import pandas as pd

file_path = os.path.join(os.path.dirname(__file__), 'sample-data/Titanic-Dataset.csv')

data = pd.read_csv(file_path)
print(data.head())