import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings

import mlflow
import mlflow.sklearn
import mlflow.xgboost

warnings.filterwarnings('ignore')

def create_confusion_matrix_plot(clf, X_test, y_test, save_path='confusion_matrix.png'):
    cmatrix = confusion_matrix(y_test, clf.predict(X_test), normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cmatrix)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(rotation=45)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_path)

if __name__ == "__main__":
    """
    This script shows how to train different classifiers on an unbalanced dataset.

    First, an unbalanced dataset is created using sklearn's make_classification function.
    Then, three classifiers are trained on the dataset: Logistic Regression, Random Forest, and XGBoost.
    The classification report is printed for each classifier.
    Finally, MLflow is used to log the models and their parameters.
    
    """
    mlflow_log_info = [] # List to store MLflow log information

    #################################
    ### Create unbalanced dataset ###
    #################################

    # Set SEEED to make experiments reproducible
    SEED = 42
    np.random.seed(SEED)

    # Create unbalanced dataset
    dataset_info = {}
    dataset_info['dataset'] = 'unbalanced'
    n_classes = 10
    total_samples = 10000
    dataset_info['n_classes'] = n_classes
    dataset_info['total_samples'] = total_samples

    class_weights = [1/n_classes] * n_classes
    class_weights = np.array(class_weights)
    # Adjust class weights to create imbalance
    class_weights[0:4] = 0.05  # Decrease weight for the first 5 classes
    class_weights[5] = 1/n_classes + 0.2
    class_weights[6:8] = 0.01
    class_weights[9] = 1/n_classes + 0.36

    X, y = make_classification(n_samples=total_samples, 
                               n_features=40,
                               n_classes=10,
                               n_clusters_per_class=5,
                               n_informative=10,
                               weights=class_weights,
                               random_state=42)
    class_id, num_samples = np.unique(y, return_counts=True)
    print(f"Class: {class_id}\nNum samples: {num_samples}\n")
    dataset_info['class_id'] = class_id
    dataset_info['num_samples'] = num_samples

    # Split the dataset into train and test sets
    test_split = 0.3
    dataset_info['test_split'] = test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=SEED)
    print(f"Train set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}\n")
    # mlflow_log_info.append(dataset_info)

    #####################################
    #### Train Classification Models ####
    #####################################
    # 1. Training Logistic Regression
    logreg_info = {}
    logreg_info['name'] = 'Logistic Regression'
    class_weights_dict = {}
    for i_clas in range(n_classes):
        class_weights_dict[i_clas] = float(1 / num_samples[i_clas])
    logreg_info['class_weights'] = class_weights_dict
    C_param = 1
    logreg_info['C'] = C_param
    log_reg = LogisticRegression(C=C_param, class_weight=class_weights_dict, solver='liblinear')
    print('Training Logistic Regression...')
    log_reg.fit(X_train, y_train)
    y_pred_log_reg = log_reg.predict(X_test)
    logreg_info['report'] = classification_report(y_test, y_pred_log_reg, output_dict=True)

    pth_conf_matrix = '../results/logreg_confusion_matrix.png'
    logreg_info['conf_matrix'] = pth_conf_matrix
    create_confusion_matrix_plot(log_reg, X_test, y_test, save_path=pth_conf_matrix)
    logreg_info['model'] = log_reg

    mlflow_log_info.append(logreg_info)

    # 2. Training Random Forest
    ranforest_info = {}
    ranforest_info['name'] = 'Random Forest'
    n_estimators = 100
    max_depth = 10
    
    ranforest_info['n_estimators'] = n_estimators
    ranforest_info['max_depth'] = max_depth
    ranforest_info['class_weights'] = class_weights_dict

    rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                class_weight=class_weights_dict)
    print('Training Random Forest Classifier...')
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    ranforest_info['report'] = classification_report(y_test, y_pred_rf, output_dict=True)

    pth_conf_matrix = '../results/rf_confusion_matrix.png'
    ranforest_info['conf_matrix'] = pth_conf_matrix
    create_confusion_matrix_plot(rf_clf, X_test, y_test, save_path=pth_conf_matrix)
    ranforest_info['model'] = rf_clf

    mlflow_log_info.append(ranforest_info)

    # 3. Training XGBoost
    xgb_info = {}
    xgb_info['name'] = 'XGBoost'
    n_estimators = 100
    xgb_info['n_estimators'] = n_estimators
    eval_metric = 'logloss'
    xgb_info['eval_metric'] = eval_metric
    max_depth = 10
    xgb_info['max_depth'] = max_depth
    xgb_clf = XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,use_label_encoder=False, eval_metric=eval_metric)
    print('Training XGBoost...')
    xgb_clf.fit(X_train, y_train)
    y_pred_xgb = xgb_clf.predict(X_test)
    xgb_info['report'] = classification_report(y_test, y_pred_xgb, output_dict=True)

    pth_conf_matrix = '../results/xg_confusion_matrix.png'
    xgb_info['conf_matrix'] = pth_conf_matrix
    create_confusion_matrix_plot(xgb_clf, X_test, y_test, save_path=pth_conf_matrix)
    xgb_info['model'] = xgb_clf

    mlflow_log_info.append(xgb_info)

    ################################
    #### Log models with MLFlow ####
    ################################
    # Set the MLflow tracking URI
    mlflow.set_experiment("Classification Models Experiment")
    # To start MLFlow server, run the following command in the terminal:
    # mlflow server --host 127.0.0.1 --port 8080
    # Copy the URL and set it to the tracking URI.
    # Alternatively paste it in the browser to see the MLflow UI
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    # Start an MLflow run
    # Log Dataset Information to MLFlow
    with mlflow.start_run(run_name='Dataset Info') as run:
        for key, val in dataset_info.items():
            mlflow.log_param(key, val)
    for i_model_info in mlflow_log_info:
        model_name = i_model_info['name']
        with mlflow.start_run(run_name=model_name) as run:
            for key, val in i_model_info.items():
                mlflow.log_param(key, val)
                if key=='conf_matrix':
                    confusion_matrix_path = val
                    mlflow.log_artifact(confusion_matrix_path, 'confusion_matrix')
                if key=='model':
                    model = val
                    if model_name == 'Logistic Regression':
                        mlflow.sklearn.log_model(model, model_name)
                    elif model_name == 'Random Forest':
                        mlflow.sklearn.log_model(model, model_name)
                    elif model_name == 'XGBoost':
                        mlflow.xgboost.log_model(model, model_name)
                if key == 'report':
                    report = val
                    mlflow.log_dict(report, 'classification_report.json')
                    for i_report in report:
                        if isinstance(report[i_report], dict):
                            for key, val in report[i_report].items():
                                mlflow.log_metric(key+f'_{i_report}', val)

    # Register MLFlow Model
    for i_model_info in mlflow_log_info:
        model_name = i_model_info['name']
        run_id = input(f"Enter the run ID for {model_name}: ")
        model_uri = f'runs:/{run_id}/{model_name}'
        mlflow.register_model(model_uri, model_name)
    print("Models logged successfully to MLFlow.")

    # Example Load Random Forest Model
    rf_model_uri = 'models:/Random Forest/1'
    rf_model = mlflow.sklearn.load_model(rf_model_uri)
    print(f'Prediction rf model: {rf_model.predict(X_test)}')