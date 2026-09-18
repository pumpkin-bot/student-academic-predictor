import pandas as pd
from src.preprocessor import DataPreprocessor

def test_preprocessor_output_dimensions():
    data = pd.DataFrame({
        "study_hours": [5.0, 2.0, 8.0, 1.0],
        "attendance": [85.0, 60.0, 95.0, 50.0],
        "past_failures": [0, 1, 0, 2],
        "internal_score": [75.0, 45.0, 88.0, 35.0],
        "internet_access": ["yes", "no", "yes", "no"],
        "target": [1, 0, 1, 0]
    })
    prep = DataPreprocessor()
    X_train, X_test, y_train, y_test = prep.fit_transform(data)
    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 1
    assert X_train.shape[1] == 5
