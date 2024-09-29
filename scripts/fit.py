# scripts/fit.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostClassifier
import yaml
import os
import joblib

# обучение модели
def fit_model():
    # Прочитайте файл с гиперпараметрами params.yaml
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    # загрузите результат предыдущего шага: initial_data.csv
    data_path = 'data/initial_data.csv'
    data = pd.read_csv(data_path)

    # Определите категориальные и числовые признаки
    cat_features = data.select_dtypes(include='object')
    potential_binary_features = cat_features.nunique() == 2

    binary_cat_features = cat_features[potential_binary_features[potential_binary_features].index]
    other_cat_features = cat_features[potential_binary_features[~potential_binary_features].index]
    num_features = data.select_dtypes(['float'])

    # Создайте трансформеры и модель
    preprocessor = ColumnTransformer(
        [
            ('binary', OneHotEncoder(drop=params['one_hot_drop']), binary_cat_features.columns.tolist()),
            ('cat', CatBoostEncoder(return_df=False), other_cat_features.columns.tolist()),
            ('num', StandardScaler(), num_features.columns.tolist())
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    model = CatBoostClassifier(auto_class_weights=params['auto_class_weights'], verbose=False)

    # Объедините всё в пайплайн
    pipeline = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )

    # Обучите модель
    pipeline.fit(data, data[params['target_col']])

    # Сохраните обученную модель
    os.makedirs('models', exist_ok=True)  # Убедитесь, что папка существует
    model_path = 'models/fitted_model.pkl'
    joblib.dump(pipeline, model_path)


if __name__ == '__main__':
    fit_model()