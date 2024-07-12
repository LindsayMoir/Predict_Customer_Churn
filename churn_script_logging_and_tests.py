"""
Test Script for Customer_Churn_Prediction
Author: Lindsay Moir
Date: July 24, 2024
"""

import logging
from churn_library import import_data, perform_eda, encoder_helper, \
    perform_feature_engineering, train_models, DATA_PATH
from logging_config import setup_logging

# Instantiate logging
LOGGING_PATH = 'logs/churn_script_logging_and_tests.log'
setup_logging(LOGGING_PATH)
logging.info("Logging has been set up successfully")

def test_import(import_data, data_path):
    """
    Test data import - Tests whether or not the file was loaded successfully.

    Parameters:
    import_data (function): Function to import data.
    data_path (str): Path to the data file.

    Returns:
    pd.DataFrame: DataFrame loaded from the data file.
    """
    try:
        df = import_data(data_path)
        logging.info("Testing import_data: SUCCESS")
        return df
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err


def test_perform_eda(perform_eda, df):
    """
    Test perform EDA function.

    Parameters:
    perform_eda (function): Function to perform exploratory data analysis.
    df (pd.DataFrame): DataFrame to test.

    Returns:
    None
    """
    try:
        df = perform_eda(df)
        assert df.isnull().sum().sum() == 0
        logging.info("Testing perform_eda: SUCCESS - No null values.")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: The DataFrame contains null values. "
            "Logistic Regression will fail with nulls."
        )
        raise err

    try:
        assert df[df.duplicated()].shape[0] == 0
        logging.info("Testing perform_eda: SUCCESS - No duplicates.")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: The DataFrame contains duplicates. "
            "Duplicates can cause problems with model results."
        )
        raise err


import logging

def test_encoder_helper(encoder_helper, df, cols):
    """
    Test encoder_helper function.

    Parameters:
    encoder_helper (function): Function to encode categorical columns.
    df (pd.DataFrame): DataFrame to test.
    cols (list): List of categorical columns to encode.

    Returns:
    pd.DataFrame: Encoded DataFrame.
    """
    expected_cols = [
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'
    ]
    
    try:
        encoded_df = encoder_helper(df, cols)

        for col in expected_cols:
            if col not in encoded_df.columns:
                raise AssertionError(
                    f"Expected column {col} not found in encoded_df DataFrame."
                )

        assert len(cols) == 5, (
            "The number of categorical columns being encoded has changed. "
            "Please check if there are any changes in the dataset or encoding process."
        )
        logging.info("Testing encoder_helper: SUCCESS")
        return encoded_df
    except AssertionError as err:
        logging.error("Testing encoder_helper: %s", err)
        raise err


def test_perform_feature_engineering(perform_feature_engineering, df, cat_columns):
    """
    Test perform_feature_engineering function.

    Parameters:
    perform_feature_engineering (function): Function to perform feature engineering.
    df (pd.DataFrame): DataFrame to test.
    cat_columns (list): List of categorical columns.

    Returns:
    tuple: X_train, X_test, y_train, y_test.
    """
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df, cat_columns)
        
        assert X_train.shape[1] == 19, (
            "The model expects 19 columns. This assertion should be changed "
            "if the number of columns changes."
        )
        
        logging.info("Testing perform_feature_engineering: SUCCESS")
        return X_train, X_test, y_train, y_test
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: %s", err)
        raise err


def test_train_models(train_models, X_train, y_train):
    """
    Test train_models function.

    Parameters:
    train_models (function): Function to train models.
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training labels.

    Returns:
    tuple: cv_rfc, lrc - Trained models.
    """
    try:
        cv_rfc, lrc = train_models(X_train, y_train)
        assert cv_rfc is not None, "Random Forest Classifier model is not trained. It is None."
        logging.info("Testing train_models: SUCCESS - Random Forest Classifier trained.")
    except AssertionError as err:
        logging.error("Testing train_models: %s", err)
        raise err

    try:
        assert lrc is not None, "Logistic Regression model is not trained. It is None."
        logging.info("Testing train_models: SUCCESS - Logistic Regression trained.")
    except AssertionError as err:
        logging.error("Testing train_models: %s", err)
        raise err


    # Run the test suite
if __name__ == "__main__":

    df = test_import(import_data, DATA_PATH)

    test_perform_eda(perform_eda, df)

    # Update with the actual categorical columns present in your dataset
    categorical_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    encoded_df = test_encoder_helper(encoder_helper, df, categorical_columns)
    X_train, X_test, y_train, y_test = test_perform_feature_engineering(
        perform_feature_engineering, encoded_df, categorical_columns)
    
    test_train_models(train_models, X_train, y_train)

    logging.info("All tests passed")
