from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ModelEvaluator:
    @staticmethod
    def evaluate(models: dict, X_test, y_test) -> dict:
        results = {}
        for name, model in models.items():
            preds = model.predict(X_test)
            results[name] = {
                "accuracy": round(accuracy_score(y_test, preds), 4),
                "precision": round(precision_score(y_test, preds), 4),
                "recall": round(recall_score(y_test, preds), 4),
                "f1_score": round(f1_score(y_test, preds), 4),
                "confusion_matrix": confusion_matrix(y_test, preds).tolist()
            }
        return results

    @staticmethod
    def select_best_model(results: dict, models: dict):
        best_name = max(results, key=lambda k: results[k]["f1_score"])
        return best_name, models[best_name]
