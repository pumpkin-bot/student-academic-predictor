from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class ModelTrainer:
    def __init__(self):
        self.models = {
            "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision_Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random_Forest": RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        }
        self.trained_models = {}

    def train_all(self, X_train, y_train):
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            self.trained_models[name] = model
        return self.trained_models
