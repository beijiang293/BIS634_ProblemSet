import pandas as pd
import os

csv_data_path = os.path.join(os.path.dirname(__file__), 'ObesityDataSet.csv')
df = pd.read_csv(csv_data_path)


# number of category in the label
number = df['NObeyesdad'].value_counts()
print(number)

label_category = ['Insufficient_Weight','Normal_Weight','Overweight_Level_I','Overweight_Level_II','Obesity_Type_I','Obesity_Type_II','Obesity_Type_III']
