#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
# cleaned_script.py
# Core model wrapper and get_models() (Chinese comments removed)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

class ModelWrapper:
    def __init__(self, model_class, param_grid, **kwargs):
        self.model_class = model_class
        self.param_grid = param_grid
        self.kwargs = kwargs
        self.best_model = None
        self.grid_search = None

    def train(self, X_train, y_train, selected_features, cv_strategy):
        base_model = self.model_class(**self.kwargs)
        self.grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.param_grid,
            cv=cv_strategy,
            scoring='roc_auc',
            n_jobs=-1,
            refit=True
        )
        self.grid_search.fit(X_train[selected_features], y_train)
        self.best_model = self.grid_search.best_estimator_
        return self.best_model

    def predict(self, X):
        if self.best_model:
            return self.best_model.predict(X)
        else:
            raise ValueError("Model has not been trained yet.")

    def predict_proba(self, X):
        if self.best_model:
            return self.best_model.predict_proba(X)
        else:
            raise ValueError("Model has not been trained yet.")

def get_models():
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt']
    }
    xgb_params = {
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [200],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0],
        'gamma': [0, 0.1],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 5],
        'eval_metric': ['logloss']
    }
    models = {
        'RandomForest': ModelWrapper(RandomForestClassifier, rf_params),
        'XGBoost': ModelWrapper(XGBClassifier, xgb_params)
    }
    return models

if __name__ == '__main__':
    print('This module defines ModelWrapper and get_models(). Import into your pipeline.')

