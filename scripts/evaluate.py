# scripts/evaluate.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import json
import yaml
import os

# Оценка качества модели
def evaluate_model():
    # Прочитайте файл с гиперпараметрами params.yaml
    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)
    
    # Загрузите данные из initial_data.csv
    data = pd.read_csv("data/initial_data.csv")
    
    # Загрузите результат прошлого шага: fitted_model.pkl
    model = joblib.load("models/fitted_model.pkl")
    
    # Реализуйте основную логику с использованием прочтённых гиперпараметров
    cv_strategy = StratifiedKFold(n_splits=params['n_splits'])
    cv_res = cross_validate(
        model,
        data.drop(columns=params['target_col']),
        data[params['target_col']],
        cv=cv_strategy,
        n_jobs=params['n_jobs'],
        scoring=params['metrics']
    )
    
    # Усредняем результаты и округляем их до 3 знаков после запятой
    cv_results = {key: round(value.mean(), 3) for key, value in cv_res.items()}
    
    # Сохраните результат кросс-валидации в cv_res.json
    os.makedirs("cv_results", exist_ok=True)
    with open("cv_results/cv_res.json", "w") as out_file:
        json.dump(cv_results, out_file)

if __name__ == '__main__':
    evaluate_model()