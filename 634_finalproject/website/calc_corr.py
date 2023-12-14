import pandas as pd
import json
import os
from sklearn.preprocessing import LabelEncoder


def calc_corr():
    csv_data_path = os.path.join(os.path.dirname(__file__), 'ObesityDataSet.csv')
    df = pd.read_csv(csv_data_path)

    columns_to_encode = [
        'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC',
        'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad'
    ]

    encoders = {}

    for column in columns_to_encode:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        encoders[column] = encoder
    corr_matrix = df.corr()

    matrix_data = corr_matrix.values.tolist()

    columns = corr_matrix.columns.tolist()
    index = corr_matrix.index.tolist()

    corr_data = {
        'matrix_data': matrix_data,
        'columns': columns,
        'index': index
    }

    corr_json = json.dumps(corr_data)
    return corr_json
