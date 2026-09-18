class StudentPredictor:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def predict_risk(self, student_features: dict) -> dict:
        processed_vec = self.preprocessor.transform_single(student_features)
        pred = self.model.predict(processed_vec)[0]
        prob = self.model.predict_proba(processed_vec)[0][pred] if hasattr(self.model, "predict_proba") else 1.0

        status = "Pass / Normal" if pred == 1 else "At-Risk of Academic Failure"
        return {
            "prediction": int(pred),
            "status": status,
            "confidence": round(float(prob), 4)
        }
