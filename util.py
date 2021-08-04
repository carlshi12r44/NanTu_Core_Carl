import inspect
import collections
import logging
from distutils.version import LooseVersion
from itertools import islice
import numpy as np

_logger = logging.getLogger(__name__)

_SAMPLE_WEIGHT = "sample_weight"

# The prefix to note that all calculated metrics and artifacts are solely based on training datasets
_TRAINING_PREFIX = "training_"


MAX_PARAMS_TAGS_PER_BATCH = 100
MAX_ENTITY_KEY_LENGTH = 250
MAX_PARAM_VAL_LENGTH = 250

# _SklearnMetric represents a metric (e.g, precision_score) that will be computed and
# logged during the autologging routine for a particular model type (eg, classifier, regressor).
_SklearnMetric = collections.namedtuple("_SklearnMetric", ["name", "function", "arguments"])

def _get_arg_names(f):
    # `inspect.getargspec` doesn't return a wrapped function's argspec
    # See: https://hynek.me/articles/decorators#mangled-signatures
    return list(inspect.signature(f).parameters.keys())

def _get_Xy(args, kwargs, X_var_name, y_var_name):
    # corresponds to: model.fit(X, y)
    if len(args) >= 2:
        return args[:2]

    # corresponds to: model.fit(X, <y_var_name>=y)
    if len(args) == 1:
        return args[0], kwargs[y_var_name]

    # corresponds to: model.fit(<X_var_name>=X, <y_var_name>=y)
    return kwargs[X_var_name], kwargs[y_var_name]

def _get_estimator_info_tags(estimator):
    """
    :return: A dictionary of MLflow run tag keys and values
             describing the specified estimator.
    """
    return {
        "estimator_name": estimator.__class__.__name__,
        "estimator_class": (estimator.__class__.__module__ + "." + estimator.__class__.__name__),
    }

def _get_sample_weight(arg_names, args, kwargs):
    sample_weight_index = arg_names.index(_SAMPLE_WEIGHT)

    # corresponds to: model.fit(X, y, ..., sample_weight)
    if len(args) > sample_weight_index:
        return args[sample_weight_index]

    # corresponds to: model.fit(X, y, ..., sample_weight=sample_weight)
    if _SAMPLE_WEIGHT in kwargs:
        return kwargs[_SAMPLE_WEIGHT]

    return None

def _get_args_for_score(score_func, fit_func, fit_args, fit_kwargs):
    """
    Get arguments to pass to score_func in the following steps.
    1. Extract X and y from fit_args and fit_kwargs.
    2. If the sample_weight argument exists in both score_func and fit_func,
       extract it from fit_args or fit_kwargs and return (X, y, sample_weight),
       otherwise return (X, y)
    :param score_func: A score function object.
    :param fit_func: A fit function object.
    :param fit_args: Positional arguments given to fit_func.
    :param fit_kwargs: Keyword arguments given to fit_func.
    :returns: A tuple of either (X, y, sample_weight) or (X, y).
    """
    score_arg_names = _get_arg_names(score_func)
    fit_arg_names = _get_arg_names(fit_func)

    # In most cases, X_var_name and y_var_name become "X" and "y", respectively.
    # However, certain sklearn models use different variable names for X and y.
    # E.g., see: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html#sklearn.multioutput.MultiOutputClassifier.fit # noqa: E501
    X_var_name, y_var_name = fit_arg_names[:2]
    Xy = _get_Xy(fit_args, fit_kwargs, X_var_name, y_var_name)

    if (_SAMPLE_WEIGHT in fit_arg_names) and (_SAMPLE_WEIGHT in score_arg_names):
        sample_weight = _get_sample_weight(fit_arg_names, fit_args, fit_kwargs)
        return (*Xy, sample_weight)

    return Xy

def _log_warning_for_metrics(func_name, func_call, err):
    msg = (
        func_call.__qualname__
        + " failed. The metric "
        + func_name
        + "will not be recorded."
        + " Metric error: "
        + str(err)
    )
    _logger.warning(msg)

def _chunk_dict(d, chunk_size):
    # Copied from: https://stackoverflow.com/a/22878842

    it = iter(d)
    for _ in range(0, len(d), chunk_size):
        yield {k: d[k] for k in islice(it, chunk_size)}

def _truncate_dict(d, max_key_length=None, max_value_length=None):
    def _truncate_and_ellipsize(value, max_length):
        return str(value)[: (max_length - 3)] + "..."

    key_is_none = max_key_length is None
    val_is_none = max_value_length is None

    if key_is_none and val_is_none:
        raise ValueError("Must specify at least either `max_key_length` or `max_value_length`")

    truncated = {}
    for k, v in d.items():
        should_truncate_key = (not key_is_none) and (len(str(k)) > max_key_length)
        should_truncate_val = (not val_is_none) and (len(str(v)) > max_value_length)

        new_k = _truncate_and_ellipsize(k, max_key_length) if should_truncate_key else k
        if should_truncate_key:
            # Use the truncated key for warning logs to avoid noisy printing to stdout
            msg = "Truncated the key `{}`".format(new_k)
            _logger.warning(msg)

        new_v = _truncate_and_ellipsize(v, max_value_length) if should_truncate_val else v
        if should_truncate_val:
            # Use the truncated key and value for warning logs to avoid noisy printing to stdout
            msg = "Truncated the value of the key `{}`. Truncated value: `{}`".format(new_k, new_v)
            _logger.warning(msg)

        truncated[new_k] = new_v

    return truncated

def _log_pretraining_metadata(estimator, *args, **kwargs):  # pylint: disable=unused-argument
        """
        Records metadata (e.g., params and tags) for a scikit-learn estimator prior to training.
        This is intended to be invoked within a patched scikit-learn training routine
        (e.g., `fit()`, `fit_transform()`, ...) and assumes the existence of an active
        MLflow run that can be referenced via the fluent Tracking API.
        :param estimator: The scikit-learn estimator for which to log metadata.
        :param args: The arguments passed to the scikit-learn training routine (e.g.,
                     `fit()`, `fit_transform()`, ...).
        :param kwargs: The keyword arguments passed to the scikit-learn training routine.
        """
        # Deep parameter logging includes parameters from children of a given
        # estimator. For some meta estimators (e.g., pipelines), recording
        # these parameters is desirable. For parameter search estimators,
        # however, child estimators act as seeds for the parameter search
        # process; accordingly, we avoid logging initial, untuned parameters
        # for these seed estimators.
        should_log_params_deeply = not _is_parameter_search_estimator(estimator)
        # Chunk model parameters to avoid hitting the log_batch API limit
        for chunk in _chunk_dict(
            estimator.get_params(deep=should_log_params_deeply),
            chunk_size=MAX_PARAMS_TAGS_PER_BATCH,
        ):
            truncated = _truncate_dict(chunk, MAX_ENTITY_KEY_LENGTH, MAX_PARAM_VAL_LENGTH)
            # try_mlflow_log(mlflow.log_params, truncated)
        # find the model parameters
        model_parameters = {}
        for key, value in estimator.__dict__.items():
            if key.endswith('_'):
                model_parameters[key] = value
        tags = _get_estimator_info_tags(estimator)
        # try_mlflow_log(mlflow.set_tags, tags)
        return {**model_parameters, **truncated, **tags}

def _log_posttraining_metadata(estimator, *args, **kwargs):
        
        """
        Records metadata for a scikit-learn estimator after training has completed.
        This is intended to be invoked within a patched scikit-learn training routine
        (e.g., `fit()`, `fit_transform()`, ...) and assumes the existence of an active
        MLflow run that can be referenced via the fluent Tracking API.
        :param estimator: The scikit-learn estimator for which to log metadata.
        :param args: The arguments passed to the scikit-learn training routine (e.g.,
                     `fit()`, `fit_transform()`, ...).
        :param kwargs: The keyword arguments passed to the scikit-learn training routine.
        """

        # try:
        #     score_args = _get_args_for_score(estimator.score, estimator.fit, args, kwargs)
        #     training_score = estimator.score(*score_args)
        # except Exception as e:
        #     msg = (
        #         estimator.score.__qualname__
        #         + " failed. The 'training_score' metric will not be recorded. Scoring error: "
        #         + str(e)
        #     )
        #     _logger.warning(msg)

        # log common metrics and artifacts for estimators (classifier, regressor)
        metrics = _log_specialized_estimator_content(estimator, args, kwargs)
        return metrics

def _is_parameter_search_estimator(estimator):
    """
    :return: `True` if the specified scikit-learn estimator is a parameter search estimator,
             such as `GridSearchCV`. `False` otherwise.
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    parameter_search_estimators = [
        GridSearchCV,
        RandomizedSearchCV,
    ]

    return any(
        [
            isinstance(estimator, param_search_estimator)
            for param_search_estimator in parameter_search_estimators
        ]
    )


def _get_metrics_value_dict(metrics_list):
    metric_value_dict = {}
    for metric in metrics_list:
        try:
            metric_value = metric.function(**metric.arguments)
        except Exception as e:
            _log_warning_for_metrics(metric.name, metric.function, e)
        else:
            metric_value_dict[metric.name] = metric_value
    return metric_value_dict



def _log_specialized_estimator_content(fitted_estimator, fit_args, fit_kwargs):
    import sklearn

    name_metric_dict = {}
    try:
        if sklearn.base.is_classifier(fitted_estimator):
            name_metric_dict = _get_classifier_metrics(fitted_estimator, fit_args, fit_kwargs)

        elif sklearn.base.is_regressor(fitted_estimator):
            name_metric_dict = _get_regressor_metrics(fitted_estimator, fit_args, fit_kwargs)
    except Exception as err:
        msg = (
            "Failed to autolog metrics for "
            + fitted_estimator.__class__.__name__
            + ". Logging error: "
            + str(err)
        )
        _logger.warning(msg)

    # if sklearn.base.is_classifier(fitted_estimator):
    #     try:
    #         artifacts = _get_classifier_artifacts(fitted_estimator, fit_args, fit_kwargs)
    #     except Exception as e:
    #         msg = (
    #             "Failed to autolog artifacts for "
    #             + fitted_estimator.__class__.__name__
    #             + ". Logging error: "
    #             + str(e)
    #         )
    #         _logger.warning(msg)
    #         return

    #     with TempDir() as tmp_dir:
    #         for artifact in artifacts:
    #             try:
    #                 display = artifact.function(**artifact.arguments)
    #                 display.ax_.set_title(artifact.title)
    #                 filepath = tmp_dir.path("{}.png".format(artifact.name))
    #                 display.figure_.savefig(filepath)
    #                 import matplotlib.pyplot as plt

    #                 plt.close(display.figure_)
    #             except Exception as e:
    #                 _log_warning_for_artifacts(artifact.name, artifact.function, e)

    #         try_mlflow_log(mlflow_client.log_artifacts, run_id, tmp_dir.path())

    return name_metric_dict

def _get_samples_labels_and_predictions(fitted_estimator, fit_args, fit_kwargs, fit_arg_names):
    # In most cases, X_var_name and y_var_name become "X" and "y", respectively.
    # However, certain sklearn models use different variable names for X and y.
    X_var_name, y_var_name = fit_arg_names[:2]
    X, y_true = _get_Xy(fit_args, fit_kwargs, X_var_name, y_var_name)
    y_pred = fitted_estimator.predict(X)

    return X, y_true, y_pred

# Util function to check whether a metric is able to be computed in given sklearn version
def _is_metric_supported(metric_name):
    import sklearn

    # This dict can be extended to store special metrics' specific supported versions
    _metric_supported_version = {"roc_auc_score": "0.22.2"}

    return LooseVersion(sklearn.__version__) >= LooseVersion(_metric_supported_version[metric_name])

def _get_classifier_metrics(fitted_estimator, fit_args, fit_kwargs):
    """
    Compute and record various common metrics for classifiers
    For (1) precision score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    (2) recall score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    (3) f1_score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    By default, we choose the parameter `labels` to be `None`, `pos_label` to be `1`,
    `average` to be `weighted` to compute the weighted precision score.
    For (4) accuracy score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    we choose the parameter `normalize` to be `True` to output the percentage of accuracy,
    as opposed to `False` that outputs the absolute correct number of sample prediction
    We log additional metrics if certain classifier has method `predict_proba`
    (5) log loss:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
    (6) roc_auc_score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    By default, for roc_auc_score, we pick `average` to be `weighted`, `multi_class` to be `ovo`,
    to make the output more insensitive to dataset imbalance.
    Steps:
    1. Extract X and y_true from fit_args and fit_kwargs, and compute y_pred.
    2. If the sample_weight argument exists in fit_func (accuracy_score by default
    has sample_weight), extract it from fit_args or fit_kwargs as
    (y_true, y_pred, ...... sample_weight), otherwise as (y_true, y_pred, ......)
    3. return a dictionary of metric(name, value)
    :param fitted_estimator: The already fitted classifier
    :param fit_args: Positional arguments given to fit_func.
    :param fit_kwargs: Keyword arguments given to fit_func.
    :return: dictionary of (function name, computed value)
    """
    import sklearn

    fit_arg_names = _get_arg_names(fitted_estimator.fit)
    X, y_true, y_pred = _get_samples_labels_and_predictions(
        fitted_estimator, fit_args, fit_kwargs, fit_arg_names
    )
    sample_weight = (
        _get_sample_weight(fit_arg_names, fit_args, fit_kwargs)
        if _SAMPLE_WEIGHT in fit_arg_names
        else None
    )

    classifier_metrics = [
        _SklearnMetric(
            name=_TRAINING_PREFIX + "precision_score",
            function=sklearn.metrics.precision_score,
            arguments=dict(
                y_true=y_true, y_pred=y_pred, average="weighted", sample_weight=sample_weight
            ),
        ),
        _SklearnMetric(
            name=_TRAINING_PREFIX + "recall_score",
            function=sklearn.metrics.recall_score,
            arguments=dict(
                y_true=y_true, y_pred=y_pred, average="weighted", sample_weight=sample_weight
            ),
        ),
        _SklearnMetric(
            name=_TRAINING_PREFIX + "f1_score",
            function=sklearn.metrics.f1_score,
            arguments=dict(
                y_true=y_true, y_pred=y_pred, average="weighted", sample_weight=sample_weight
            ),
        ),
        _SklearnMetric(
            name=_TRAINING_PREFIX + "accuracy_score",
            function=sklearn.metrics.accuracy_score,
            arguments=dict(
                y_true=y_true, y_pred=y_pred, normalize=True, sample_weight=sample_weight
            ),
        ),
    ]

    if hasattr(fitted_estimator, "predict_proba"):
        y_pred_proba = fitted_estimator.predict_proba(X)
        classifier_metrics.extend(
            [
                _SklearnMetric(
                    name=_TRAINING_PREFIX + "log_loss",
                    function=sklearn.metrics.log_loss,
                    arguments=dict(y_true=y_true, y_pred=y_pred_proba, sample_weight=sample_weight),
                ),
            ]
        )

        if _is_metric_supported("roc_auc_score"):
            # For binary case, the parameter `y_score` expect scores must be
            # the scores of the class with the greater label.
            if len(y_pred_proba[0]) == 2:
                y_pred_proba = y_pred_proba[:, 1]

            classifier_metrics.extend(
                [
                    _SklearnMetric(
                        name=_TRAINING_PREFIX + "roc_auc_score",
                        function=sklearn.metrics.roc_auc_score,
                        arguments=dict(
                            y_true=y_true,
                            y_score=y_pred_proba,
                            average="weighted",
                            sample_weight=sample_weight,
                            multi_class="ovo",
                        ),
                    ),
                ]
            )

    return _get_metrics_value_dict(classifier_metrics)

def _get_regressor_metrics(fitted_estimator, fit_args, fit_kwargs):
    """
    Compute and record various common metrics for regressors
    For (1) (root) mean squared error:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    (2) mean absolute error:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
    (3) r2 score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
    By default, we choose the parameter `multioutput` to be `uniform_average`
    to average outputs with uniform weight.
    Steps:
    1. Extract X and y_true from fit_args and fit_kwargs, and compute y_pred.
    2. If the sample_weight argument exists in fit_func (accuracy_score by default
    has sample_weight), extract it from fit_args or fit_kwargs as
    (y_true, y_pred, sample_weight, multioutput), otherwise as (y_true, y_pred, multioutput)
    3. return a dictionary of metric(name, value)
    :param fitted_estimator: The already fitted regressor
    :param fit_args: Positional arguments given to fit_func.
    :param fit_kwargs: Keyword arguments given to fit_func.
    :return: dictionary of (function name, computed value)
    """
    import sklearn

    fit_arg_names = _get_arg_names(fitted_estimator.fit)
    _, y_true, y_pred = _get_samples_labels_and_predictions(
        fitted_estimator, fit_args, fit_kwargs, fit_arg_names
    )
    sample_weight = (
        _get_sample_weight(fit_arg_names, fit_args, fit_kwargs)
        if _SAMPLE_WEIGHT in fit_arg_names
        else None
    )

    regressor_metrics = [
        _SklearnMetric(
            name=_TRAINING_PREFIX + "mse",
            function=sklearn.metrics.mean_squared_error,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight,
                multioutput="uniform_average",
            ),
        ),
        _SklearnMetric(
            name=_TRAINING_PREFIX + "mae",
            function=sklearn.metrics.mean_absolute_error,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight,
                multioutput="uniform_average",
            ),
        ),
        _SklearnMetric(
            name=_TRAINING_PREFIX + "r2_score",
            function=sklearn.metrics.r2_score,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight,
                multioutput="uniform_average",
            ),
        ),
    ]

    # To be compatible with older versions of scikit-learn (below 0.22.2), where
    # `sklearn.metrics.mean_squared_error` does not have "squared" parameter to calculate `rmse`,
    # we compute it through np.sqrt(<value of mse>)
    metrics_value_dict = _get_metrics_value_dict(regressor_metrics)
    metrics_value_dict[_TRAINING_PREFIX + "rmse"] = np.sqrt(
        metrics_value_dict[_TRAINING_PREFIX + "mse"]
    )

    return metrics_value_dict