################################
###### Import Modules ##########
################################

import importlib
import settings
importlib.reload(settings)

# import data frameworks
import pandas as pd
import numpy as np

# import viz
import matplotlib.pyplot as plt
import seaborn as sns

# import ML
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as Pipeline_imb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
from scipy.stats import kurtosis, skew
from imblearn.over_sampling import SMOTENC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, make_scorer



# import others
import os
import pickle
import warnings


################################
###### Notebook Settings  ######
################################

pd.set_option('display.max_columns', None) 


################################
###### Global Variable #########
################################

DATA_RAW_DIR = "data_raw"
DATA_RAW_FILE = "Sleep_health_and_lifestyle_dataset.csv"
DATA_EDA_DIR = "data_eda"
DATA_EDA_FILE = "Sleep_health_and_lifestyle_dataset_eda.csv"
ALL_FEATURES = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 
                'Physical Activity Level', 'Stress Level', 'BMI Category', 'Heart Rate', 
                'Daily Steps', 'Blood Pressure Category']
LABEL = ['Sleep Disorder']

# MINIMUM_TRACKING_QUARTERS = 4
# TARGET = "foreclosure_status"
# NON_PREDICTORS = [TARGET, "id"]
# CV_FOLDS = 3


################################
###### Functions Defs ##########
################################


def examine_values(dataframe):
    """
    Generate a summary of unique values for each column in the given DataFrame.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing unique values for each column.
    """
    dict = {}
    for col in dataframe.columns:
        unique_values = dataframe[col].unique()
        dict[col] = [unique_values]

    result = pd.DataFrame(dict).transpose()
    result.columns = ["Unique Values"]
    pd.set_option('display.max_colwidth', 1000)
    return result

def column_mapper(dataframe, column, map):
    """
    Map values in a specific column of the DataFrame using the provided mapping.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to be mapped.
        map (dict): A dictionary containing the mapping of old values to new values.
    """
    dataframe[column] = dataframe[column].replace(map)

def numerical_univariate_summary(dataframe, numerical_columns):
    """
    Generate a summary of numerical univariate statistics for the given columns in the DataFrame.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        numerical_columns (list): List of column names containing numerical data.

    Returns:
        pd.DataFrame: A DataFrame containing various statistics for each numerical column.
    """
    dict = {}
    for col in numerical_columns:
        desc = dataframe[col].describe()
        count = desc['count']
        mean = desc['mean']
        std = desc['std']
        min_val = desc['min']
        q1 = desc['25%']
        median = desc['50%']
        q3 = desc['75%']
        max_val = desc['max']
        kurtosis_val = kurtosis(dataframe[col])
        skewness = skew(dataframe[col])
        outlier_flag = ((max_val > q3 + (1.5 * (q3 - q1))) | (min_val < q1 - (1.5 * (q3 - q1))))
        
        dict[col] = {
            "count": count, "mean": mean, "std": std, "min": min_val,
            "q1": q1, "median": median, "q3": q3, "max": max_val,
            "kurtosis": kurtosis_val, "skewness": skewness, "outlier_flag": outlier_flag
        }

    result = pd.DataFrame(dict).transpose()
    pd.set_option('display.max_colwidth', 1000)
    return result

def categorical_univariate_summary(dataframe, categorical_columns):
    """
    Generate a summary of categorical univariate statistics for the given columns in the DataFrame.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        categorical_columns (list): List of column names containing categorical data.

    Returns:
        pd.DataFrame: A DataFrame containing various statistics for each categorical column.
    """
    dict = {}
    for col in categorical_columns:
        distinct_count = dataframe[col].nunique()
        value_counts = dataframe[col].value_counts()
        dict[col] = {"distinct_count": distinct_count, "value_counts": value_counts}
        
    result = pd.DataFrame(dict).transpose()
    pd.set_option('display.max_colwidth', 1000)
    return result


def categorical_roll_up(dataframe, column, new_category_name, threshold):
    """
    Roll up infrequent categories in a categorical column into a new category.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        column (str): The name of the categorical column.
        new_category_name (str): The name of the new category to replace infrequent ones.
        threshold (int): The threshold below which categories will be rolled up.
    """
    count_data = dataframe[column].value_counts()
    to_replace = count_data[count_data < threshold].index
    # Replace the labels
    dataframe[column] = dataframe[column].replace(to_replace, new_category_name)


def fix_columns(df, columns):
    """
    Ensure that the DataFrame has the specified columns, adding missing ones if needed.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to ensure in the DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with ensured columns.
    """
    missing_cols = set(columns) - set(df.columns)
    if len(missing_cols) > 0:
        print("Columns from train added: ", missing_cols)
    for col in missing_cols:
        df[col] = 0  
    # Ensure all required columns are present
    assert set(columns) - set(df.columns) == set()
    extra_cols = set(df.columns) - set(columns)
    if extra_cols:
        print("extra columns from test removed: ", extra_cols)
    df = df[columns]
    return df



# Define a function to convert transformed array to DataFrame
def convert_transformed_features_to_df(ColumnTransformer, transformed_array):
    """
    Converts a transformed array to a DataFrame with appropriate column names. 
    Rquirements:
        ColumnTransformer object must have been used to fit the transformed_array. 
        ***All columns must have been transformed. i.e. Remainder must not have been used.***
        ***This function should be revisted to allow for remainder to be used.***

    Args:
        pipeline (Pipeline): The pipeline containing the transformers.
        transformed_array (numpy.ndarray): The transformed data array.
        
    Returns:
        pandas.DataFrame: DataFrame with transformed features and column names.
    """
    # steps = list(ColumnTransformer.named_transformers_.keys())

    # feature_names = []

    # for step in [step for step in steps if step != "remainder"]:
    #     # Get the feature names after transformation from the pipeline
    #     step_feature_names = ColumnTransformer.named_transformers_[step].get_feature_names_out().tolist()
    #     feature_names.extend(step_feature_names)

    # Get column names of transformed array
    feature_names = ColumnTransformer.get_feature_names_out()

    feature_names = [feature_names.split("__")[1] for feature_names in feature_names]
    
    # Create a DataFrame using the transformed array and column names
    transformed_df = pd.DataFrame(transformed_array, columns=feature_names)
    
    return transformed_df


def create_pipeline(*steps_list):
    """
    Creates a pipeline composed of preprocessing and machine learning model steps.
    
    Args:
        *steps_list: Variable number of two-element lists where each list contains the step name and the step object.
        
    Returns:
        pipeline: A scikit-learn pipeline with specified preprocessing and model steps.
    """
    # Create a list of tuples with step names and step objects
    steps = [(step[0], step[1]) for step in steps_list]
    
    # Create the pipeline using the defined steps
    pipeline = Pipeline(steps)
    
    # Return the created pipeline
    return pipeline



def save_pipeline(file_name, pipeline_to_save):
    """
    Save the provided pipeline to a file.

    Args:
        file_name (str): The filename to save the pipeline.
        pipeline_to_save (Pipeline): The pipeline to be saved.

    Returns:
        None
    """
    # Save the provided pipeline to the specified file
    with open(file_name, 'wb') as pipeline_file:
        pipeline = pickle.dump(pipeline_to_save, pipeline_file)



def get_saved_pipeline(file_name):
    """
    Load and return a pipeline from a saved file.

    Args:
        file_name (str): The filename of the saved pipeline.

    Returns:
        Pipeline: The loaded pipeline.
    """
    # Load the best current pipeline from the file
    with open(file_name, 'rb') as pipeline_file:
        loaded_best_current_pipeline = pickle.load(pipeline_file)
    
    return loaded_best_current_pipeline


def score_formatter(score, precision):
    """
    Format a score as a percentage string.

    Args:
        score (float): The score to be formatted.
        precision (int): The number of decimal places for precision.

    Returns:
        str: The formatted score as a percentage string.
    """
    formatted_score = f"{np.round(score * 100, precision):.{precision}f} %"
    return formatted_score



def plot_conf_matrix(class_labels, confusion_matrix):
    """
    Plot a confusion matrix as a heatmap.

    Args:
        class_labels (list): List of class labels for the x and y axis tick labels.
        confusion_matrix (array-like): The confusion matrix to be plotted.

    Returns:
        None
    """
    # Create a heatmap for the confusion matrix
    plt.subplots(figsize=(6, 4), gridspec_kw={'hspace': 0.8}, facecolor="#F3EEE7")
    heatmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    
    # Add plot details
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
    plt.gca().xaxis.tick_top()  # Place x-axis ticks on top for better alignment
    plt.show()



def plot_micro_averaged_roc(fpr_micro, tpr_micro, roc_auc_micro=None):
    """
    Plot the micro-averaged ROC curve.

    Args:
        fpr_micro (array-like): False positive rates for the micro-averaged ROC curve.
        tpr_micro (array-like): True positive rates for the micro-averaged ROC curve.
        roc_auc_micro (float or None): The AUC value for the micro-averaged ROC curve.

    Returns:
        None
    """
    plt.subplots(figsize=(6, 4), gridspec_kw={'hspace': 0.8}, facecolor="#F3EEE7")
    
    if roc_auc_micro is not None:
        # Plot micro-averaged ROC curve with AUC label
        plt.plot(fpr_micro, tpr_micro, label=f'Micro-Average (AUC = {roc_auc_micro:.2f})')
    else:
        # Plot micro-averaged ROC curve without AUC label
        plt.plot(fpr_micro, tpr_micro)
    
    # Add plot details
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-Averaged ROC Curve')
    
    if roc_auc_micro is not None:
        plt.legend(loc="lower right")
    
    plt.show()


def get_gridsearchcv_summary(fitted_grid_search, precision, X_test, y_test):
    """
    Display a summary of the results from a fitted GridSearchCV object.

    Parameters:
        fitted_grid_search (GridSearchCV): A fitted GridSearchCV object after hyperparameter tuning.
        precision (int): Number of decimal places to display for accuracy scores.
        X_test (array-like): Test data features.
        y_test (array-like): Test data labels.
    """

    # Access best hyperparameters and accuracy score
    best_params = fitted_grid_search.best_params_
    best_train_score = fitted_grid_search.best_score_

    # Print the best parameters
    print("Best Parameters:", pd.DataFrame(best_params, index=[0]))

    # Print the best mean cv training score
    print("Mean CV Train Accuracy with best parameters:", score_formatter(best_train_score, precision))

    # Evaluate the best model on the test set
    best_pipeline = fitted_grid_search.best_estimator_
    test_accuracy = best_pipeline.score(X_test, y_test)
    print("Test Accuracy with best parameters:", score_formatter(test_accuracy, precision))


def plot_seed_variability(X, y, test_size, current_seed, num_seeds, pipeline_or_model, scoring, cv_object, pipeline_type="sklearn"):
    """
    Evaluate and visualize the variability of model performance across different random seeds.

    This function splits the provided dataset into train and test subsets using different random seeds.
    It then trains a given model or pipeline on the training data, computes train and test accuracies,
    as well as cross-validation scores. The function produces a plot showcasing the trend of these scores
    across different seeds.

    Parameters:
    X (array-like): The feature matrix.
    y (array-like): The target vector.
    test_size (float): The proportion of the dataset to include in the test split.
    num_seeds (int): The number of random seeds to consider.
    pipeline_or_model: The machine learning model or pipeline to evaluate.
    scoring (str): The scoring metric for cross-validation and accuracy evaluation.
    cv_object: The cross-validation strategy, compatible with the `cross_val_score` function.

    Returns:
    None
        The function prints the average train, cross-validation, and test accuracy scores across seeds,
        and displays a plot illustrating the performance variability.
    """
    train_cv_dict = {}
    train_dict = {}
    test_dict = {}
    
    for i in np.arange(1, num_seeds + 1):
        # re-split using seed
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i, stratify=y)
        # Reshape y vector
        y_train, y_test = y_train.values.reshape(-1), y_test.values.reshape(-1)
        # run training cv score and add to dict
        cv_scores = cross_val_score(pipeline_or_model, X_train, y_train, scoring=scoring, cv=cv_object)
        train_cv_dict[i] = cv_scores.mean()
        # fit pipeline
        pipeline_or_model.fit(X_train, y_train)
        # score on train and add to dict
        train_accuracy = pipeline_or_model.score(X_train, y_train)
        train_dict[i] = train_accuracy
        # score on test and to dict
        allowed_pipeline_type = ["sklearn", "imblearn"]
        if pipeline_type not in allowed_pipeline_type:
            raise ValueError(f"'pipeline_type' must be one of {allowed_pipeline_type}")
        elif pipeline_type == "sklearn":
            test_accuracy = pipeline_or_model.score(X_test, y_test)
        else:
            steps = pipeline_or_model.steps # collect all steps of pipeline
            transformer_count = len(steps) - 2 # collect indicies of non-model and non-smote steps
            transformers = [step[1] for step in steps[:transformer_count]] # collect ordered transformer objects
            model = steps[-1][1]  # collect model
            X_test_transformed = X_test
            for transformer in transformers:
                X_test_transformed = transformer.transform(X_test_transformed)
            test_accuracy = model.score (X_test_transformed, y_test)
            train_dict[i] = train_accuracy
        test_dict[i] = test_accuracy

    # calculate average train and test accuracy scores across all seeds
    ave_train = np.mean(list(train_dict.values()))
    ave_train_cv = np.mean(list(train_cv_dict.values()))
    ave_test = np.mean(list(test_dict.values()))

    # present averages to screen
    print("Average train score across seeds:", settings.score_formatter(ave_train,3))
    print("Average cv train score across seeds:", settings.score_formatter(ave_train_cv,3))
    print("Average test score across seeds:", settings.score_formatter(ave_test,3))

    # plot results
    fig, ax = plt.subplots(figsize=(6, 4), gridspec_kw={'hspace': 0.8}, facecolor="#F3EEE7")
    ax.plot(np.arange(1, num_seeds + 1), train_dict.values(), color="r", alpha=0.4, label="Train Accuracy")
    ax.axhline(y=np.mean(list(train_dict.values())), color='r', linestyle='dotted')
    ax.plot(np.arange(1,num_seeds + 1), train_cv_dict.values(), color="b", alpha=0.4, label="Test CV Accuracy")
    ax.axhline(y=np.mean(list(train_cv_dict.values())), color='b', linestyle='dotted')
    ax.plot(np.arange(1,num_seeds + 1), test_dict.values(), color="g", alpha=0.4, label="Test Accuracy")
    ax.axhline(y=np.mean(list(test_dict.values())), color='g', linestyle='dotted')
    ax.axvline(x=current_seed, color='black', linestyle='solid', alpha=0.4)
    ax.axvline(x=min_seed, color='black', linestyle='dotted', alpha=0.4)
    ax.set_ylabel(scoring.capitalize())
    ax.set_xlabel("Seed")
    ax.set_title("Train, Train CV and Test Score Variability Across Seeds")
    ax.legend()
    plt.show()
  
    # return seed to original
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=current_seed, stratify=y)




def hyperparam_tune_summary(search_object, X_train, y_train, X_test, y_test, scorer):
    """
    Summarizes the results of hyperparameter tuning using GridSearchCV.

    Args:
        search_object (GridSearchCV): GridSearchCV object containing the results of hyperparameter tuning.
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        X_test (array-like): Test data features.
        y_test (array-like): Test data labels.

    Returns:
        None (prints the summary to the console).
    """
    
    # Access best hyperparameters and accuracy score
    best_train_cv_params, best_train_cv_score = search_object.best_params_, search_object.best_score_

    # Print the best parameters
    print("Best Parameters:")
    display(pd.DataFrame(best_train_cv_params, index=[0]))

    # Get best pipeline
    best_pipeline = search_object.best_estimator_

    print("---------------------")
    print("Using best parameters")
    print("---------------------")
    
    # Fit pipeline
    best_pipeline.fit(X_train, y_train)

    # predict on pipeline
    y_pred_train = best_pipeline.predict(X_train)
    y_pred_test = best_pipeline.predict(X_test)

    # score pipeline on train
    train_score = scorer(y_train, y_pred_train)
    print('Train Score:', settings.score_formatter(train_score, 3))
    
    # Print CV train score
    # Access the cv_results_ dictionary from the GridSearchCV object
    cv_results = search_object.cv_results_

    # Find the index of the best estimator in the cv_results_ arrays
    best_index = np.argmax(cv_results['rank_test_score'] == 1)

    # Extract the cross-validation scores for the best estimator
    best_train_cv_std = cv_results['std_test_score'][best_index]

    print('Mean CV Train Score:', settings.score_formatter(best_train_cv_score, 3),
        "( +-", settings.score_formatter(best_train_cv_std, 3), ")")

    # score pipeline on test
    test_score = scorer(y_test, y_pred_test)
    print('Test Score:', settings.score_formatter(test_score, 3))



def cross_val_summary(pipeline, cv_scores, X_train, y_train, X_test, y_test, scorer):
    """
    Calculate and print a summary of cross-validation and test accuracy for a given pipeline.

    Parameters:
    pipeline (Pipeline): The pipeline to be evaluated.
    cv_scores (numpy.ndarray): Array of cross-validation scores.
    X_train (numpy.ndarray): Training data features.
    y_train (numpy.ndarray): Training data labels.
    X_test (numpy.ndarray): Test data features.
    y_test (numpy.ndarray): Test data labels.
    """
    
    # Fit pipeline
    pipeline.fit(X_train, y_train)

    # predict on pipeline
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    # score pipeline on train
    train_score = scorer(y_train, y_pred_train)
    print('Train Score:', settings.score_formatter(train_score, 3))

    # Calculate mean and standard deviation of cross-validation train scores
    mean_train_cv_score = np.mean(cv_scores)
    std_train_cv_score  = np.std(cv_scores)
    print('Mean CV Train Score:', settings.score_formatter(mean_train_cv_score, 4),
          "( +-", settings.score_formatter(std_train_cv_score, 3), ")")

    # score pipeline on test
    test_score = scorer(y_test, y_pred_test)
    print('Test Score:', settings.score_formatter(test_score, 3))



################################
########## Class Defs ##########
################################


# create binning transformer class
class CustomBinnerNB(BaseEstimator, TransformerMixin):
    # def __init__(self)
    #     # print("\n>>>>>>>>init() called.\n")
    
    def fit(self, X, y = None):
        # print("\n>>>>>>>>fit() called.\n")
        return self

    def transform(self, X, y = None):
        # print("\n>>>>>>>>transform() called.\n")
        X_bin = X.copy()
        # Bin age to new column and drop original column
        X_bin.loc[:,'Age Bin'] = pd.cut(X_bin.loc[:,'Age'], 
        [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, float('inf')], 
        labels=['0-4', '5-9', '10-15', '10-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '>100']).astype("object")
        X_bin = X_bin.drop("Age", axis=1)

        # Bin sleep duration to new column and drop original column
        X_bin.loc[:,'Sleep Duration Bin'] = pd.cut(X_bin.loc[:,'Sleep Duration'], 
        [0, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, float('inf')], 
        labels=['<4', '4.0-4.4', '4.5-4.9', '5.0-5.4', '5.5-5.9', '6.0-6.4', '6.5-6.9', '7.0-7.4', '7.5-7.9', '8.0-8.4', '8.5-8.9', '9.0-9.4', '9.5-9.9', '>10']).astype("object")
        X_bin = X_bin.drop("Sleep Duration", axis=1)

        # Bin physical activity level to new column and drop original column
        X_bin.loc[:,'Physical Activity Level Bin'] = pd.cut(X_bin.loc[:,'Physical Activity Level'], 
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')], 
        labels=['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99', '>100']).astype("object")
        X_bin = X_bin.drop("Physical Activity Level", axis=1)
    
        # Bin heart rate to new column and drop original column
        X_bin.loc[:,'Heart Rate Bin'] = pd.cut(X_bin.loc[:,'Heart Rate'], 
        [0, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, float('inf')], 
        labels=['<40', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '>100']).astype("object")
        X = X.drop("Heart Rate", axis=1)

        # Bin daily steps to new column and drop original column
        X_bin.loc[:,'Daily Steps Bin'] = pd.cut(X_bin.loc[:,'Daily Steps'], 
        [0, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 100000, float('inf')], 
        labels=['<3000', '3000-3499', '3500-3999', '4000-4499', '4500-4999', '5000-5499', '5500-5999', '6000-6499', '6500-6999', '7000-7499', '7500-7999', '8000-8499', '8500-8999', '9000-9499', '9500-9999', '>100']).astype("object")
        X_bin = X_bin.drop("Daily Steps", axis=1)

        X_bin = X_bin.astype(str)

        return X_bin



class CustomPositiveF1Scorer():
    def __init__(self):
        # Define the scorer function
        def positive_f1_scorer(y_true, y_pred):
            return f1_score(y_true, y_pred, labels=[1, 2], average='micro')

        # Assign the scorer function to self.scorer_function
        self.scorer_function = positive_f1_scorer

        # Create the custom scorer using make_scorer
        self.custom_scorer = make_scorer(self.scorer_function)