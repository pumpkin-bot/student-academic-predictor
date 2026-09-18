from src.data_loader import get_or_create_data
from src.preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.predictor import StudentPredictor

def run_pipeline():
    print("=== Step 1: Loading / Synthesizing Dataset ===")
    df = get_or_create_data("data/student_data.csv")
    print(f"Dataset ready with {len(df)} records.")

    print("\n=== Step 2: Preprocessing Features ===")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

    print("\n=== Step 3: Training Multiple Models ===")
    trainer = ModelTrainer()
    trained_models = trainer.train_all(X_train, y_train)

    print("\n=== Step 4: Model Evaluation ===")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(trained_models, X_test, y_test)
    for model_name, score in metrics.items():
        print(f"[{model_name}] -> Accuracy: {score['accuracy']}, F1-Score: {score['f1_score']}")

    best_name, best_model = evaluator.select_best_model(metrics, trained_models)
    print(f"\nOptimal Selected Model: {best_name}")

    print("\n=== Step 5: Test Single Prediction ===")
    predictor = StudentPredictor(best_model, preprocessor)
    sample_student = {
        "study_hours": 2.5,
        "attendance": 55.0,
        "past_failures": 2,
        "internal_score": 42.0,
        "internet_access": "yes"
    }
    result = predictor.predict_risk(sample_student)
    print(f"Sample Student Input: {sample_student}")
    print(f"Risk Evaluation Result: {result['status']} (Confidence: {result['confidence']})")

if __name__ == "__main__":
    run_pipeline()
