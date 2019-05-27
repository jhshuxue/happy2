#!/usr/bin/env python

import numpy as np
from sklearn.externals import joblib
import xgboost as xgb


def get_sepsis_score(data, model):
	
    data1=data[-1].reshape(1,-1)
    dfull=xgb.DMatrix(data1)
    score=model.predict(dfull,validate_features=False)
    label=score>0.5
    return score, label


def load_sepsis_model():
	clf = joblib.load("bst_v2.pkl")
	return clf
