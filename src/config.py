# --- modulos
import os
import logging

# --- paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(DATA_DIR, 'models')
DATA_SPLIT_FILE = os.path.join(DATA_DIR, 'train_test.joblib')

# --- models
MODEL_FILES = {
    'MLP': os.path.join(MODEL_DIR, 'mlp.joblib'),
    'SVM': os.path.join(MODEL_DIR, 'svm.joblib'),
    'DecisionTree': os.path.join(MODEL_DIR, 'dt.joblib'),
    'RandomForest': os.path.join(MODEL_DIR, 'rf.joblib')
}

# --- features order
FEATURE_ORDER = ['income', 'age', 'loan']

# --- logging
LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOGGING_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# --- riscos
RISK_THRESHOLDS = {
    'low_moderate': 70, 
    'moderate_high': 50 
}