import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

csv_data_path = os.path.join(os.path.dirname(__file__), 'ObesityDataSet.csv')
columns_to_encode = [
    'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC',
    'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad'
]


def load_data(csv_data_path):
    return pd.read_csv(csv_data_path)


def preprocess_data(df, columns_to_encode):
    encoders = {}

    for column in columns_to_encode:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        encoders[column] = encoder

    return df, encoders


def get_encode_dict(encoders):
    encode_dict = {}
    for col, encoder in encoders.items():
        encode_dict[col] = {}
        each_col_original_labels = encoder.classes_
        each_col_encoded_labels = encoder.transform(each_col_original_labels)

        for original, encoded in zip(each_col_original_labels, each_col_encoded_labels):
            encode_dict[col][encoded] = original
    return encode_dict


def split_data(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)  # 特征
    y = df[target_column]  # 标签
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, y_train, n_neighbors=7):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def predict_knn(model, scaler, features, feature_names):
    features_df = pd.DataFrame([features], columns=feature_names)
    features_scaled = scaler.transform(features_df)
    prediction = model.predict(features_scaled)


    return prediction[0]


def loading():

    df = load_data(csv_data_path)
    df, encoders = preprocess_data(df, [
        'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC',
        'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad'
    ])
    encode_dict = get_encode_dict(encoders)
    X_train, X_test, y_train, y_test = split_data(df, 'NObeyesdad')
    X_train_scaled, X_test_scaled, scaler = standardize_data(X_train, X_test)

    model = train_model(X_train_scaled, y_train)
    return model, scaler, encode_dict


if __name__ == "__main__":
    model, scaler, _ = loading()
    feature_name = [
        "Gender", "Age", "Height", "Weight", "family_history_with_overweight",
        "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC",
        "FAF", "TUE", "CALC", "MTRANS"
    ]

    result = predict_knn(model, scaler, features=[1, 22, 1.75, 85, 1, 1, 3, 3, 1, 0, 2, 0, 0.5, 1, 3, 0],
                         feature_names=feature_name)
    print(result)
