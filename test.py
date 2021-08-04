import intercepts
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import base64
import json
import requests
from requests.exceptions import ConnectionError
import os
import logging
import pathlib
from sklearn import metrics
from sklearn.metrics import average_precision_score, mean_squared_error, r2_score
import inspect
from util import _log_pretraining_metadata, _log_posttraining_metadata

_logger = logging.getLogger(__name__)

NTCORE_WORKSPACE_ID = 'NTCORE_WORKSPACE_ID'
NTCORE_HOST = os.getenv('NTCORE_HOST') if os.getenv('NTCORE_HOST') is not None else 'http://localhost:8180'
BASE_URL = '{ntcore_host}/dsp/api/v1/workspace/{workspace_id}/experiment'
os.environ["DSP_RUNTIME_VERSION"] = "python-3.7"
os.environ["FRAMEWORK"] = "sklearn"

def set_workspace_id(workspace_id):
    """
    Sets global workspace id.
    """
    os.environ[NTCORE_WORKSPACE_ID] = workspace_id

def _get_workspace_id():
    """
    Retrieves workspace id.
    """
    try:
        workspace_id = os.environ[NTCORE_WORKSPACE_ID]
        return workspace_id
    except Exception as e:
        pass

    try:
        workDir = pathlib.Path().absolute()
        relative_path = os.path.relpath(workDir, os.environ['DSP_INSTANCE_HOME'])
        return relative_path.split('/')[0]
    except Exception as e:
        _logger.warning('Unable to read workspace id from work dir: {0}.'.format(e))
        return None

def _get_runtime_version():
    """
    Retrieves runtime version.
    """
    try:
        runtime_version = os.environ["DSP_RUNTIME_VERSION"]
        return runtime_version
    except Exception as e:
        _logger.warning("Please set env variable 'DSP_RUNTIME_VERSION'. Acceptable values are 'python-3.7', 'python-3.8' and 'python-3.9'.")
        return
"""Old handler version"""
# def handler(func, *args, **kwargs):
#     """
#     handler for intercepting functions
    
#     Args:
#         func: function for intercept
#         *args: arguments
#         **kwargs: key arguments
#     Returns:
#         model_parameters: the parameters the model used 
#         model_hyper_paramters: hyper parameters the model used 

#         TODO 07/29/2021
#         1. 参考 _get_args_for_score in mlflow/sklearn  和 _log_specialized_estimator_content in mlflow/sklearn 来算metrics
#         2. 参考 _get_estimator_info_tags in line 806 mlflow/sklearn 
#     """
#     model = func(*args, **kwargs)
#     model_parameters = {}
#     for key, value in model.__dict__.items():
#         if key.endswith('_'):
#             model_parameters[key] = value
#     model_hyper_parameters = model.get_params()
#     print(f"in intercepts I am receiving the parameters from model {model} as {model_parameters}")
#     print(f"in intercepts I am receiving the hyper parameters from model {model} as {model_hyper_parameters}")
    
#     return model_parameters, model_hyper_parameters
"""New handler version"""
def handler(func, *args, **kwargs):
    """
    handler for intercepting functions
    
    Args:
        func: function for intercept
        *args: arguments
        **kwargs: key arguments
    Returns:
        model_parameters: the parameters the model used 
        model_hyper_paramters: hyper parameters the model used 

        TODO 07/29/2021
        1. 参考 _get_args_for_score in mlflow/sklearn  和 _log_specialized_estimator_content in mlflow/sklearn 来算metrics
        2. 参考 _get_estimator_info_tags in line 806 mlflow/sklearn 
    """
    model = func(*args, **kwargs)
    model_parameters = _log_pretraining_metadata(model)
    
    return model_parameters

def _preprocess(dict):
    """
    preprocess paramters

    """
    ans = {}
    for k,v in dict.items():
        if type(v) is np.ndarray:
            ans[k] = v.tolist()
        else:
            ans[k] = v
    return ans
def _send_model_to_ntcore(estimator, framework, parameters, metrics):
    """
    Sends metrics to ntcore server.
    INPUTS:
        estimator: model itself
        framework: machine learning framework, in this case, sklearn
        paramters: model paramters 
        metrics: model metrics 

    OUTPUTS: 
        post request to the server

    """
    # runtime is 3.9 
    # framework -> sklearn
    # model -> estimator (SVC() for example) use pickle to dump the model and use base64.encode(pickcle ) use pickle transfer model file into byte format
    # createExperimentV1, emit_model_to_ntcore in  is the reference 
    workspace_id = _get_workspace_id()
    
    if workspace_id is None:
        _logger.warning("This experiment is not logged because no workspace id is found. You can use mflow.set_workspace_id(...) to set the workspace id.")
    

    model_blob = pickle.dumps(estimator)
    parameters_converted = _preprocess(parameters)
    metrics_converted = _preprocess(metrics)
    endpoint = BASE_URL.format(ntcore_host=NTCORE_HOST, workspace_id=workspace_id)
    payload = { "runtime": _get_runtime_version(),
                "framework": framework,
                "parameters": json.dumps(parameters_converted),
                "metrics": json.dumps(metrics_converted), 
                "model": base64.b64encode(model_blob) }

    try:
        requests.post(endpoint, data=payload)
    except ConnectionError as connectionError:
         _logger.warning('This experiment is not logged in ntcore server because ntcore server is not running.')


if __name__=="__main__":
    set_workspace_id("CEII4VPOFNA164BVT3JYZWDUBK")
    ## example of sklearn https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    
    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()
    # register through intercepts 
    intercepts.register(regr.fit, handler)
    # Train the model using the training sets (New Handler version)
    model_parameters = regr.fit(diabetes_X_train, diabetes_y_train)
    # metrics after training
    _metrics = _log_posttraining_metadata(regr, diabetes_X_train, diabetes_y_train)
    # make the post request (See createExperimentV1) TODO
    _send_model_to_ntcore(regr, os.environ["FRAMEWORK"], model_parameters, _metrics)
 
    intercepts.unregister_all()