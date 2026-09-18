```markdown
# Problem Statement: Student Academic Risk & Early Intervention System

## Problem Statement
Higher education institutions often struggle to identify students at risk of academic probation or failure before semester-end examinations. Interventions deployed after grades are finalized are ineffective. A predictive model capable of flagging at-risk students using early attendance, study hours, assignment completion rates, and internal assessment trends allows academic counselors to stage targeted interventions.

## Scope of the Project
The project encompasses:
- Synthetic/historical feature ingestion (study hours, past failures, internal exam marks, attendance).
- Automated feature cleaning, categorical encoding, and standard scaling.
- Training and comparative analysis across Logistic Regression, Decision Tree, and Random Forest classifiers.
- Evaluation on classification metrics (Accuracy, Precision, Recall, F1-Score).
- A standalone inference pipeline supporting real-time CLI assessment.

## Target Users
- Academic Advisors & Counselors: For early warnings and scheduling 1-on-1 tutoring.
- Course Instructors: To track cohort-level engagement indicators.
- Institutional Administrators: For retention trend analysis.

## High-Level Features
1. Automated Pipeline Ingestion: Generates or parses student historical records.
2. Multi-Model Benchmark: Trains and ranks models; exports the best-performing model.
3. Risk Scoring Engine: Evaluates new student parameters and flags Pass vs At-Risk.
