# Predict Customer Churn
"""
This program predicts customer churn. It is implemented in a modular way to expedite the
use of test suites. It also logs activities to make it easier to run and debug in production.

Author: Lindsay Moir
Creation Date: July 12, 2024
"""

# Import Libraries
from logging_config import setup_logging
import joblib
import warnings
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, RocCurveDisplay, classification_report, \
    precision_score, recall_score, accuracy_score, f1_score
import logging
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# Set the QT_QPA_PLATFORM environment variable to 'offscreen'.
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# config data

LOG_FILE_PATH = 'logs/churn_modelling_metrics_log.csv'  # For modelling run metrics
LOGGING_PATH = 'logs/churn_library.log'  # For logging of each function
DATA_PATH = "data/bank_data.csv"  # Path to the data

# for the coefficients of the features realted to churn
COEF_LIST_PATH = 'artifacts/coef_list.txt'

# Saved model paths
MODEL_RF_PATH = "models/rfc_model.pkl"
MODEL_LR_PATH = "models/logistic_model.pkl"


def import_data(path):
    """
    Import the data from the path provided and return a pandas dataframe.

    Parameters:
    path (str): The file path to the data.

    Returns:
    pd.DataFrame: The data loaded into a pandas DataFrame.

    Raises:
    FileNotFoundError: If the file at the provided path does not exist.
    """
    try:
        df = pd.read_csv(path, index_col=0)
        logging.info("SUCCESS: import_eda: Loading %s", path)
    except FileNotFoundError as err:
        logging.error("Error: import_eda: The file wasn't found")
        raise err

    return df


def perform_eda(df):
    """
    Perform EDA on the data and return the cleaned data.

    Parameters:
    df (pd.DataFrame): The input data.

    Returns:
    pd.DataFrame: The cleaned data.
    """
    # Check the size
    if df.shape[0] > 1000 and df.shape[1] > 20:
        pass
    else:
        logging.error("ERROR: Data does not meet the size requirements.")

    # Check for nulls
    if df.isnull().sum().sum() == 0:
        pass
    else:
        logging.error("ERROR: There are nulls. Logistic Regression will fail.")

    # For console output
    print(df.describe())

    # Check for duplicates
    dups = df[df.duplicated()]
    if dups.shape[0] == 0:
        pass
    else:
        logging.error(
            "ERROR: There are duplicates. Duplicate cleaning code does not exist in this program.")

    # Create a churn column to enable further analysis and then drop that
    # column
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df.drop(columns=["Attrition_Flag"], inplace=True)

    return df


# EDA Plots
def eda_plots(df, coef_list_path):
    """
    Create EDA plots and save them as artifacts.

    Parameters:
    df (pd.DataFrame): The input data.
    coef_list_path (str): The path to save the coefficients list.
    """
    # Produce plots and save as artifacts
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig('images/eda/Churn_hist.png')

    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig('images/eda/Customer_Age_hist.png')

    plt.figure(figsize=(20, 10))
    df['Marital_Status'].value_counts(normalize=True).plot(kind='bar')
    plt.savefig('images/eda/Marital_Status_bar.png')

    # Suppress all warnings
    warnings.filterwarnings("ignore")
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('images/eda/Total_Trans_Ct_hist.png')
    # Turn warnings back on:
    warnings.filterwarnings("default")

    # Run some correlations
    # Get the numeric columns
    numeric_df = df.select_dtypes(include=np.number)

    plt.figure(figsize=(20, 10))
    sns.heatmap(numeric_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('images/eda/Correlation_Heat_Map.png')

    # We are predicting Churn. Let's look at that in particular
    coef_list = [(df['Churn'].corr(numeric_df[col]), col)
                 for col in numeric_df]
    coef_list.sort(reverse=True)

    # Write the coefficients to the text file
    with open(coef_list_path, 'w', encoding='utf-8') as file:
        for coef in coef_list:
            file.write(f"{coef}\n")

    logging.info("SUCCESS: Completed writing %s", coef_list_path)


# Encoder Helper
def encoder_helper(df, cols):
    """
    Create new columns in the dataframe that show the churn rate for each
    category in the categorical columns.

    Parameters:
    df (pd.DataFrame): The input data.
    cols (list): List of categorical columns to encode.

    Returns:
    pd.DataFrame: The dataframe with new encoded columns.
    """
    for col in cols:
        groups = df[[col, 'Churn']].groupby(col).mean()['Churn']
        df[col + '_Churn'] = df[col].map(groups)

    return df


def perform_feature_engineering(df, cat_columns):
    """
    Perform feature engineering and return the train and test data. Since we are encoding
    the actual churn rate into the new quantitative columns called 'Categorical_Column_Churn',
    we will need to make sure that there is no data leakage between the train and test data.
    This will require us to run encoder_helper AFTER we have done the train test split.

    Parameters:
    df (pd.DataFrame): The input data.
    cat_columns (list): List of categorical columns to encode.

    Returns:
    tuple: X_train, X_test, y_train, y_test dataframes.
    """
    # Create X and y
    y = df['Churn']
    X = df.drop(columns=['Churn', 'CLIENTNUM'])

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Concatenate the dataframes back together now that you have the train
    # test split
    X_train['Churn'] = y_train
    X_test['Churn'] = y_test

    # Run encoder_helper
    X_train = encoder_helper(X_train, cat_columns)
    X_test = encoder_helper(X_test, cat_columns)

    # Drop the dependent variable from X_train and X_test
    X_train.drop(columns=['Churn'], inplace=True)
    X_test.drop(columns=['Churn'], inplace=True)

    # Keep only quantitative columns
    numeric_columns = X_train.select_dtypes(include=['number'])

    # Update X_train and X_test to include only the numeric columns
    X_train = X_train[numeric_columns.columns]
    X_test = X_test[numeric_columns.columns]

    # Write all of the files to disk
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)

    return X_train, X_test, y_train, y_test


# Train Models
def train_models(X_train, y_train):
    """
    Train each model and return the best model.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training labels.

    Returns:
    tuple: Trained RandomForestClassifier and LogisticRegression models.
    """
    # Grid search for RandomForestClassifier
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1
    )
    cv_rfc.fit(X_train, y_train)

    # Train and fit Logistic Regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(X_train, y_train)

    return cv_rfc, lrc


# Scores
def scores(
    model_name,
    y_test,
    y_test_preds,
    y_train,
    y_train_preds,
    log_file_path
):
    """
    Calculate the scores and write them to a log file.

    Parameters:
    model_name (str): Name of the model.
    y_test (pd.Series): True labels for the test set.
    y_test_preds (pd.Series): Predicted labels for the test set.
    y_train (pd.Series): True labels for the train set.
    y_train_preds (pd.Series): Predicted labels for the train set.
    log_file_path (str): Path to the log file.

    Returns:
    None
    """
    # Scores
    print(f'{model_name} Results')

    print('Test results')
    print('KPI Metric ROC_AUC Score:', roc_auc_score(y_test, y_test_preds))
    print(classification_report(y_test, y_test_preds))

    print('Train results')
    print('KPI Metric ROC_AUC Score:', roc_auc_score(y_train, y_train_preds))
    print(classification_report(y_train, y_train_preds))

    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_test_preds)
    precision = precision_score(y_test, y_test_preds)
    recall = recall_score(y_test, y_test_preds)
    accuracy = accuracy_score(y_test, y_test_preds)
    f1 = f1_score(y_test, y_test_preds)

    # Write metrics to .csv
    metrics_data = {
        'Timestamp': [datetime.now()],
        'Model Name': [model_name],
        'ROC AUC': [roc_auc],
        'Precision': [precision],
        'Recall': [recall],
        'Accuracy': [accuracy],
        'F1-score': [f1]
    }

    # Write results to log file
    metrics_df = pd.DataFrame(metrics_data)

    try:
        if os.path.exists(log_file_path):
            # Append to existing file
            metrics_df.to_csv(
                log_file_path,
                mode='a',
                header=False,
                index=False
            )
            logging.info("Metrics appended to log_runs.csv")
        else:
            # Create a new file
            metrics_df.to_csv(log_file_path, index=False)
            print("Metrics saved to log_runs.csv\n")
    except IOError as e:
        logging.error("Error occurred writing %s: %s", log_file_path, str(e))

    if roc_auc < 0.9 and model_name == 'Random Forest':
        logging.error(
            "ERROR: The %s ROC_AUC score at %s is < .9., model_name, roc_auc.")
    else:
        logging.info(
            "SUCCESS: The %s ROC_AUC score is %s",
            model_name,
            roc_auc)


# Plots From Modelling
def plots_from_modelling(cv_rfc, lrc, X_test, y_test):
    """
    Create plots from the modelling results.

    Parameters:
    cv_rfc (GridSearchCV): Trained RandomForestClassifier with GridSearchCV.
    lrc (LogisticRegression): Trained LogisticRegression model.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test labels.

    Returns:
    None
    """
    # Logistic Regression ROC Curve plot
    lrc_plot = RocCurveDisplay.from_estimator(lrc, X_test, y_test)
    plt.savefig('images/results/lrc_plot_auc.png')

    # Random Forest and Logistic Regression AUC plot
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8
    )
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('images/results/rfc_lrc_plot_auc.png')


# Classification Report Image
def classification_report_image(
        name,
        y_train,
        y_train_preds,
        y_test,
        y_test_preds):
    """
    Create a classification report image.

    Parameters:
    name (str): Name of the model.
    y_train (pd.Series): True labels for the training set.
    y_train_preds (pd.Series): Predicted labels for the training set.
    y_test (pd.Series): True labels for the test set.
    y_test_preds (pd.Series): Predicted labels for the test set.

    Returns:
    None
    """
    # Classification report for the model
    plt.rc('figure', figsize=(5, 5))

    plt.text(
        0.01, 1.25, f'{name} Train', {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds)), {
             'fontsize': 10}, fontproperties='monospace')

    plt.text(
        0.01, 0.6, f'{name} Test', {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds)), {
             'fontsize': 10}, fontproperties='monospace')

    plt.axis('off')
    path = f'images/results/Classification_Report_{name}.png'
    plt.savefig(path)
    plt.close()


# Feature Importance
def feature_importance(cv_rfc, X_train, X_test):
    """
    Calculate the feature importance and create a plot.

    Parameters:
    cv_rfc (GridSearchCV): Trained RandomForestClassifier with GridSearchCV.
    X_train (pd.DataFrame): Training features.
    X_test (pd.DataFrame): Test features.

    Returns:
    None
    """
    # Shap plot
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig('images/results/shap_tree_explainer.png')

    # Calculate feature importances
    importances = cv_rfc.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_train.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_train.shape[1]), importances[indices])
    plt.xticks(range(X_train.shape[1]), names, rotation=90)

    plt.savefig('images/results/feature_importance.png')


# Driver
def driver(path, log_file_path, coef_list_path, model_rf_path, model_lr_path):
    """
    Run the entire modelling process.

    Parameters:
    path (str): Path to the input data file.
    log_file_path (str): Path to the log file.
    coef_list_path (str): Path to save the coefficient list.
    model_rf_path (str): Path to save the Random Forest model.
    model_lr_path (str): Path to save the Logistic Regression model.

    Returns:
    None
    """
    # Load data
    df = import_data(path)
    logging.info("SUCCESS: Read file %s", path)

    # Perform EDA
    df = perform_eda(df)
    logging.info("SUCCESS: Completed perform_eda")

    # EDA plots
    eda_plots(df, coef_list_path)
    logging.info("SUCCESS: Completed eda_plots")

    # Perform feature engineering and deal with the categorical columns
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df, df.select_dtypes(include=['object']).columns
    )
    logging.info(
        "SUCCESS: perform_feature_engineering completed %s, %s, %s, %s",
        X_train.shape, X_test.shape, y_train.shape, y_test.shape
    )

    # Suppress all warnings
    warnings.filterwarnings("ignore")

    # Train and fit both models
    cv_rfc, lrc = train_models(X_train, y_train)
    logging.info("SUCCESS: Completed train_models")

    # Turn warnings back on
    warnings.filterwarnings("default")

    # Predict using both models
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    logging.info("SUCCESS: Completed Random Forest predictions")

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    logging.info("SUCCESS: Completed Logistic Regression predictions")

    # Scores
    scores(
        'Random Forest',
        y_test,
        y_test_preds_rf,
        y_train,
        y_train_preds_rf,
        log_file_path
    )
    scores(
        'Logistic Regression',
        y_test,
        y_test_preds_lr,
        y_train,
        y_train_preds_lr,
        log_file_path
    )
    logging.info(
        "SUCCESS: Completed scoring the models and writing the log file to disk.")

    # Save best model
    joblib.dump(cv_rfc.best_estimator_, model_rf_path)
    joblib.dump(lrc, model_lr_path)

    # Prove they loaded ok
    rfc_model = joblib.load(model_rf_path)
    if rfc_model:
        logging.info("SUCCESS: Completed reloading the Random Forest model.")
    lr_model = joblib.load(model_lr_path)
    if lr_model:
        logging.info(
            "SUCCESS: Completed reloading the Logistic Regression model.")

    # Plots from modelling
    plots_from_modelling(cv_rfc, lrc, X_test, y_test)
    logging.info("SUCCESS: Completed plots_from_modelling.")

    # Classification reports
    classification_report_image(
        'Logistic Regression',
        y_train,
        y_train_preds_lr,
        y_test,
        y_test_preds_lr
    )
    classification_report_image(
        'Random Forest',
        y_train,
        y_train_preds_rf,
        y_test,
        y_test_preds_rf
    )
    logging.info("SUCCESS: Completed classification_report_image.")

    # Feature importance
    feature_importance(cv_rfc, X_train, X_test)
    logging.info("SUCCESS: Completed feature_importance.")


if __name__ == "__main__":

    # Instantiate logging
    setup_logging(LOGGING_PATH)
    logging.info("SUCCESS: Started run at %s.", datetime.now())

    driver(DATA_PATH,
           LOG_FILE_PATH,
           COEF_LIST_PATH,
           MODEL_RF_PATH,
           MODEL_LR_PATH)

    logging.info(
        "SUCCESS: Completed entire modelling run at %s!",
        datetime.now())
