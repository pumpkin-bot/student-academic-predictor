import numpy as np
from src.model_trainer import ModelTrainer

def test_model_training():
    X_train = np.array([[5.0, 80.0, 0, 70.0, 1], [1.0, 50.0, 2, 35.0, 0]] * 10)
    y_train = np.array([1, 0] * 10)

    trainer = ModelTrainer()
    trained = trainer.train_all(X_train, y_train)

    assert "Random_Forest" in trained
    preds = trained["Random_Forest"].predict(X_train)
    assert len(preds) == 20
