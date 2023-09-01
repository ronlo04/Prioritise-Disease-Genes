import os
from google.cloud import bigquery
import json
import pandas as pd
import subprocess
import networkx as nx
import pickle
import numpy as np
import matplotlib.pyplot as plt
from node2vec import Node2Vec
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pulearn import BaggingPuClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, make_scorer, average_precision_score
from sklearn.model_selection import train_test_split
import seaborn as sns


def combine_json_splits (json_files_path, output_file_path):
    '''
    Utlise json library to read each json splits
    and use Pandas to integrate the data and export to a single JSON file
    parameters: 
        json_files_path: string, the path contain the json files
        output_file_path: string, the path to store the processed files
    return:
        None
    exception:
        JSON Decode error
    '''
    data = []
    for filename in os.listdir(json_files_path):
        if filename.endswith('.json'):
            file_path = os.path.join(json_files_path, filename)
            with open(file_path, 'r') as file:
                for line in file:
                    try:
                        json_data = json.loads(line)
                        data.append(json_data)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file: {filename}")
                        print(e)

    df = pd.DataFrame(data)
    df.to_json(output_file_path)
    
    return None

def get_asso_data_google_big_query (api_key_path, disease_id, datatype_id, output_file_name_path):
    '''
    use google big query API to make query to the OpenTargets Database, and use pandas to process the download data and export it to a csv file
    
    parameters:
        api_key_path: str, the path where the api key is stored
        disease_id: str, the id of the disease
        datatype_id: str, the id of the datatype
        output_file_name_path: str, the path where te output file is stored
    return:
        None
    '''
    # set up the api client
    client = bigquery.Client.from_service_account_json(api_key_path)

    # prepare the queries
    query_statement = f"""
        SELECT
        associations.targetId AS target_id,
        targets.approvedSymbol AS target_approved_symbol,
        associations.diseaseId AS disease_id,
        diseases.name AS disease_name,
        associations.score AS genetic_association_score_Indirect_and_direct
        FROM
        `open-targets-prod.platform.associationByDatatypeIndirect` AS associations
        JOIN
        `open-targets-prod.platform.diseases` AS diseases
        ON
        associations.diseaseId = diseases.id
        JOIN
        `open-targets-prod.platform.targets` AS targets
        ON
        associations.targetId = targets.id
        WHERE
        associations.diseaseId= '{disease_id}'
        and
        associations.datatypeId='{datatype_id}'
        ORDER BY
        associations.score DESC
    """
    genetic_asso_df = client.query(query_statement).to_dataframe()
    genetic_asso_df.to_csv(output_file_name_path, index=False)

    return None

def has_self_loop(graph):
    '''
    check if there is self interacting nodes
    parameter: 
        graph: networkx graph object
    return 
        True: if there is self-interacting node
        False: if there is no self-interacting node
    '''
    for node in graph.nodes():
        if node in graph.neighbors(node):
            return True
    return False

def nodes_with_self_loop(graph):
    '''
    get the nodes that is self-interacting
    parameter: 
        graph: networkx graph object
    return: 
        list, a list of nodes that are self-interacting
    '''
    nodes_with_loop = []
    for node in graph.nodes():
        if node in graph.neighbors(node):
            nodes_with_loop.append(node)
    return nodes_with_loop

def plot_log_log_degree_dist(graph):
    '''
    plot the log log degree distribution from the graph object
    parameter: 
        G: networkx graph object
    return: 
        list, a list of nodes that are self-interacting
    '''
    degree_of_each_node = [v for k,v in nx.degree(graph)]
    uq_degree_vals = sorted(set(degree_of_each_node))
    frequency_for_each_uq_degree_val = [degree_of_each_node.count(x) for x in uq_degree_vals]

    x = np.asarray(uq_degree_vals, dtype = float)
    y = np.asarray(frequency_for_each_uq_degree_val, dtype = float)

    logx = np.log10(x)
    logy = np.log10(y)

    plt.figure(figsize=(10,10))
    plt.xlim(min(logx), max(logx))
    plt.xlabel('log10 (Degree)')
    plt.ylabel('log10 (Number of nodes)')
    plt.title('Degree Distribution of Network')
    out_degree_dist = plt.plot(logx, logy, 'o')
    
    return None

# define helper function to parse embedding output of node2vec
def load_embedding(file_path_and_name):
    '''
    read the embedding file which is export from the node2vec. parse the embedding and output as a pandas dataframe
    parameter:
        file_path_and_name: str, the embedding file including the path
    return: 
        df: pandas dataframe object
    '''
    with open(file_path_and_name, 'r') as file:
        lines = file.readlines()

    num_rows, num_features = map(int, lines[0].split())

    data = []

    for line in lines[1:]:
        values = line.strip().split()
        data.append(values)

    df = pd.DataFrame(data)

    column_names = ['id'] + ['feature_' + str(i) for i in range(1, num_features+1)]
    df.columns = column_names

    return df

# define helper function to positive labelling data
def positive_labelling (data, positive_ids):
    '''
    label positive data should the key match with the key in the data object. 1 for positive class, 0 the otherwise
    the labelled column is 'y'
    
    parameters:
        data: pandas dataframe, the target dataset
        positive_ids: pandas dataframe, the list of positive id
    retrun
        data: pandas dataframe, a dataframe stacked with a column 'y' as the class label of the data
    '''
    data = data.copy()
    def set_y_value(row):
        if row['id'] in positive_ids:
            return 1
        else:
            return 0

    data['y'] = data.apply(set_y_value, axis=1)

    return data

def evaluate_models(models, models_name, train_X, train_y, test_X, test_y, scorers_with_args, eval_models_path, eval_results_path):
    '''
    - train the list of models from the given training dataset and evaluate the predictions performance on the given testing data,
    - score the performance by the input scorers
    - and export the trained models and the performance results to the designated path
    
    parametes:
        models: list of BaggingPUClassifier objects, models for evaluation
        models_name: string, the common name of the model list
        train_X: pandas dataframe object, training data
        train_y: pandas dataframe object, true lable of training data
        test_X: pandas dataframe object, testing data
        test_y: pandas dataframe object, true lable of testing data
        scorers_with_args: list of tuples. Each tuple consists of scorer name, scoring function, and the keyword args
        eval_models_path: string, the path to store the trained models
        eval_results_path: string, the path to store the score of the respective scores
    return:
        None
    '''
    results = []

    for model in models:
        model.fit(train_X, train_y)

    eval_models_file_name = models_name + '.pickle'
    eval_models_file_path = os.path.join(eval_models_path, eval_models_file_name)

    with open(eval_models_file_path, 'wb') as f:
        pickle.dump(models, f, pickle.HIGHEST_PROTOCOL)

    for model in models:
        result_for_a_model = evaluate(model, test_X, test_y, scorers_with_args)
        print(result_for_a_model)
        results.append(result_for_a_model)

    results_df = pd.DataFrame(results)
    eval_results_file_name = 'eval_result_' + models_name + '.csv'
    eval_results_file_path = os.path.join(eval_results_path, eval_results_file_name)

    results_df.to_csv(eval_results_file_path, index=False)
    
def get_avg_feature_importances (models):
    '''
    Iterate all Bagging classifers in the model list. For example if the bagging classifier is Random Forest, 
    this function obtain the feature importance from the decision trees in the emsemble, 
    and average the feature importance of each random forest model, 
    and then average the feature importance of all random forest models. 
    The base classifier of the input models must implement feature_importance method. 
    
    parametes:
        models: list of BaggingPUClassifier objects, models that's base classifier implemented feature_importance method. For example, base classifier is Decision Tree
    return:
        sorted_means: pandas dataframe, sorted order of the feature name and its importance value [0.0-1.0] in descending order 
    '''

    all_rf_importances = []

    for idx, model in enumerate(models):
        clf = model.named_steps['clf']
        feature_importances = [estimator.feature_importances_ for estimator in clf.estimators_]
        df_feature_importances = pd.DataFrame(feature_importances)
        feature_importances_means = df_feature_importances.mean()
        all_rf_importances.append(feature_importances_means)

    # Convert list of feature importances to a DataFrame
    df_all_rf_importances = pd.concat(all_rf_importances, axis=1).T

    # Calculate the mean feature importance across all models
    average_importance = df_all_rf_importances.mean()

    # Sort by descending importance
    sorted_means = average_importance.sort_values(ascending=False)
    
    return sorted_means

def map_proba(dataset, y_pred_proba, all_id=True):
    '''
    map the predicted probability to the unlabelled genes
    parameters:
        dataset: pandas dataframe, the dataset that the final model is produced
        y_pred_proba: numpy array, the predictions being make on each unlabelled data
        all_id: boolean, if the return should include all records or only the unlabbelled data
    return:
        output: pandas dataframe, the prediction results mapped with the dataset id
    '''
    data = dataset.copy()
    data['y_pred_proba'] = np.nan
    
    # create a boolean mask for unlabelled data rows where 'y' is 0
    unlab_data_mask = data['y'] == 0

    # ensure the length of y_pred_proba matches the number of True values in the mask
    assert len(y_pred_proba) == sum(unlab_data_mask), "Length mismatch between predictions and mask."

    # apply predictions
    data.loc[unlab_data_mask, 'y_pred_proba'] = y_pred_proba
    data = data[['id', 'y_pred_proba']]
    
    if not all_id:
        data = data.loc[unlab_data_mask]
        
    return data