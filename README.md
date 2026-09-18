# Student Academic Risk & Performance Predictor

An end-to-end machine learning system designed to detect students at risk of academic failure early in the semester, enabling automated tracking and timely counselor intervention.

## Architectural Workflow
```
[Raw Student Data] 
       │
       ▼
[Data Preprocessor] ──> (Encoding & StandardScaler)
       │
       ▼
[Model Benchmark]   ──> (Logistic Regression, Decision Tree, Random Forest)
       │
       ▼
[Evaluator Engine]  ──> (Accuracy, Precision, Recall, F1 Selection)
       │
       ▼
[Inference Service] ──> (Single/Batch Student Academic Risk Output)
```

## Functional Modules
- Data Ingestion Module: Reads CSV inputs or synthesizes standard student feature distributions.
- Preprocessing Pipeline: Implements categorical mapping, missing-value fallbacks, and feature scaling.
- Model Training Benchmarking: Concurrently fits linear and non-linear classifiers.
- Evaluation & Selection: Generates confusion matrices and selects the highest F1-performing model.
- Risk Inference Engine: Generates confidence-rated predictions for target profiles.

## Non-Functional Requirements
- Performance: Model inference executes in under 15ms per student instance.
- Maintainability: Modular architecture split into single-responsibility Python files.
- Reliability: Input schema validation prevents pipeline failures on unseen values.
- Testability: Unit test suite integrated via PyTest.

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/pumpkin-bot/student-academic-predictor.git](https://github.com/pumpkin-bot/student-academic-predictor.git)
   cd student-academic-predictor
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application
```bash
python main.py
```

## Running Automated Tests
```bash
pytest
```
