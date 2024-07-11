"""Test Script for Customer_Churn_Prediction
"""

import logging
from churn_library import import_data, perform_eda, encoder_helper, \
    perform_feature_engineering, train_models, DATA_PATH
from logging_config import setup_logging

# Instantiate logging
LOGGING_PATH = 'logs/churn_script_logging_and_tests.log'
setup_logging(LOGGING_PATH)


def test_import(import_data, DATA_PATH):
    '''churn_script_logging_and_tests
    Test data import - Tests whether or not the file was loaded successfully.
    '''
    try:
        df = import_data(DATA_PATH)
        logging.info("Testing import_data: SUCCESS")
        # Flush the handler to ensure all buffered log messages are written to
        # file
        logging.getLogger().handlers[0].flush()
        return df
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err


def test_perform_eda(perform_eda, df):
    '''
    Test perform EDA function
    '''
    try:
        df = perform_eda(df)
        assert df.isnull().sum().sum() == 0
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: The file has nulls. Logistic Regression will fail with nulls.")
        raise err

    try:
        assert df[df.duplicated()].shape[0] == 0
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: The file has duplicates. Duplicates cause problems with model \
                results.")
        raise err


def test_encoder_helper(encoder_helper, df, cols):
    '''
    Test encoder_helper
    '''
    expected_cols = [
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'
    ]
    try:
        encoded_df = encoder_helper(df, cols)

        for col in expected_cols:
            if col not in encoded_df.columns:
                raise AssertionError(
                    f"Expected column {col} not found in encoded_df DataFrame.")

        assert len(cols) == 5, "The number of categorical columns being encoded has changed. \
            Please check if there are any changes in the dataset or encoding process."
        logging.info("Testing encoder_helper: SUCCESS")
        return encoded_df
    except AssertionError as err:
        logging.error("Testing encoder_helper: %s", err)
        raise err


def test_perform_feature_engineering(
        perform_feature_engineering, df, cat_columns):
    '''
    Test perform_feature_engineering
    '''
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df, cat_columns)
        assert X_train.shape[1] == 19, "The model expects 19 columns. This assertion should \
            be changed if the number of columns changes."
        logging.info("Testing perform_feature_engineering: SUCCESS")
        return X_train, X_test, y_train, y_test
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The model expects 19 columns. This \
                assertion should be changed if the number of columns changes.")
        raise err


def test_train_models(train_models, X_train, y_train):
    '''
    Test train_models
    '''
    try:
        cv_rfc, lrc = train_models(X_train, y_train)
        assert cv_rfc is not None
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing train_models: The model is not trained. It is None")
        raise err

    try:
        assert lrc is not None
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing train_models: The model is not trained. It is None")
        raise err


if __name__ == "__main__":

    # Run the test suite
    
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
