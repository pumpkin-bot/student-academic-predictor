import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None

    def fit_transform(self, df: pd.DataFrame):
        df_encoded = self._encode_categorical(df.copy())
        X = df_encoded.drop(columns=["target"])
        y = df_encoded["target"]
        self.feature_columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def transform_single(self, input_dict: dict):
        df = pd.DataFrame([input_dict])
        df_encoded = self._encode_categorical(df)
        for col in self.feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[self.feature_columns]
        return self.scaler.transform(df_encoded)

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        if "internet_access" in df.columns:
            df["internet_access"] = df["internet_access"].map({"yes": 1, "no": 0}).fillna(0)
        return df
