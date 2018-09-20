"""
Actually, This is a Simple and Humble Framework for Machine Learning powered by a CZK
=====================================================================================

Just import this framework like this:
Copy this file to the filefolder and then use this command:
>>> hyper = __import__("Hyper-Framework")

Load the dataset as a pandas.DataFrame before predicting:
>>> import pandas as pd
>>> iris = pd.read_csv( filepath_or_buffer, names = [ "sepal-length", "sepal-width", "petal-length", "petal-width" ] )

And then use the fuctions in it:
>>> y_Predicted = _HYPER_PREDICT(iris)

Enjoy Yourself!
"""



import numpy as np, \
       scipy as sp

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use("seaborn")

import sklearn

from sklearn import model_selection, \
                    preprocessing, \
                    pipeline, \
                    metrics

from sklearn.externals import joblib

from sklearn.linear_model import LogisticRegression, \
                                 LinearRegression, \
                                 Ridge, \
                                 Lasso, \
                                 ElasticNet

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier, \
                         DecisionTreeRegressor, \
                         ExtraTreeClassifier, \
                         ExtraTreeRegressor

from sklearn.neighbors import KNeighborsClassifier, \
                              KNeighborsRegressor

from sklearn.svm import SVC, \
                        SVR

from sklearn.ensemble import GradientBoostingClassifier, \
                             GradientBoostingRegressor, \
                             RandomForestClassifier, \
                             RandomForestRegressor, \
                             ExtraTreesClassifier, \
                             ExtraTreesRegressor, \
                             AdaBoostClassifier, \
                             AdaBoostRegressor

import sklearn.datasets as SKLEARN_DATASETS



N_SPLITS = 10
RANDOM_STATE = 7
TEST_SIZE = 0.2
SCORING = "accuracy"
PLOTTING_SWITCH = True
SUPPORTED_MODELS_TO_GRID = [
    KNeighborsClassifier, KNeighborsRegressor
    # SVC, SVR
    ]



def _AUTO_SPLITTING_DATASET( _Data, test_size = TEST_SIZE, random_state = RANDOM_STATE ): # Fine!
    return model_selection.train_test_split(
        _Data.values[ : , : (-1) ],
        _Data.values[ : , (-1) ],
        test_size = test_size,
        random_state = random_state
        )



def _GET_ORDINARY_MODEL_DICT(_Type = "classification"): # Fine!
    if _Type == "classification":
        return {
            "LR" : LogisticRegression(),
            "LDA" : LinearDiscriminantAnalysis(),
            "GNB" : GaussianNB(),
            "KNC" : KNeighborsClassifier(),
            "DTC" : DecisionTreeClassifier(),
            "ETC" : ExtraTreeClassifier(),
            "SVC" : SVC()
            }
    elif _Type == "regression":
        return {
            "LR" : LinearRegression(),
            "RIDGE" : Ridge(),
            "LASSO" : Lasso(),
            "EN" : ElasticNet(),
            "KNR" : KNeighborsRegressor(),
            "DTR" : DecisionTreeRegressor(),
            "ETR" : ExtraTreeRegressor(),
            "SVR" : SVR()
            }
    else:
        raise Exception("")

def _GET_ENSEMBLE_MODEL_DICT(_Type = "classification"): # Fine!
    if _Type == "classification":
        return {
            "GBC" : GradientBoostingClassifier(),
            "RFC" : RandomForestClassifier(),
            "ETC" : ExtraTreesClassifier(),
            "ABC" : AdaBoostClassifier()
            }
    elif _Type == "regression":
        return {
            "GBR" : GradientBoostingRegressor(),
            "RFR" : RandomForestRegressor(),
            "ETR" : ExtraTreesRegressor(),
            "ABR" : AdaBoostRegressor()
            }
    else:
        raise Exception("")



def _ALGO_CMP( Models, X, y, _Scale = False, _Plot = False, _CVType = "KFold", n_splits = N_SPLITS, random_state = RANDOM_STATE, scoring = SCORING, test_size = TEST_SIZE ): # Fine!

    _Results = []

    if _Scale:
        _Scaler = preprocessing.StandardScaler().fit(X = X)
        X = _Scaler.transform(X = X)

    if _CVType == "KFold":
        _Cross_Val = model_selection.KFold( n_splits = n_splits, random_state = random_state )
    elif _CVType == "LeaveOneOut":
        _Cross_Val = model_selection.LeaveOneOut()
    elif _CVType == "ShuffleSplit":
        _Cross_Val = model_selection.ShuffleSplit( n_splits = n_splits, test_size = test_size, random_state = random_state )
    else:
        raise Exception()

    for _Each in Models:
        _CVResult = model_selection.cross_val_score( estimator = Models[_Each], X = X, y = y, scoring = SCORING, cv = _Cross_Val )
        _Results.append(( _Each, _CVResult ))

    if _Plot:
        plt.title("Comparison")
        plt.boxplot( x = [ _Results[i][1] for i in range(len(_Results)) ], labels = Models.keys() )
        plt.show()

    _Best_Model = _Results[0]
    for i in range( 1, len(_Results) ):
        if _Results[i][1].mean() > _Best_Model[1].mean():
            _Best_Model = _Results[i]
        elif _Results[i][1].mean() == _Best_Model[1].mean():
            if _Results[i][1].std() > _Best_Model[1].std():
                _Best_Model = _Results[i]
    _Best_Model = Models[_Best_Model[0]]

    return _Results, _Best_Model



def _MODEL_RUN( Model, X_TRAIN, X_TEST, y_TRAIN, y_TEST, _Scale = False, _Type = "classification" ): # Fine!

    if _Scale:
        _Scaler = preprocessing.StandardScaler().fit(X = X_TRAIN)
        X_TRAIN = _Scaler.transform(X = X_TRAIN)
        X_TEST = _Scaler.transform(X_TEST)

    Model.fit( X_TRAIN, y_TRAIN )
    y_PRED = Model.predict(X_TEST)

    if _Type == "classification":
        _Metrics = {
            "ACCURACY_SCORE" : metrics.accuracy_score( y_TEST, y_PRED ),
            "CONFUSION_MATRIX" : metrics.confusion_matrix( y_TEST, y_PRED ),
            "CLASSIFICATION_REPORT" : metrics.classification_report( y_TEST, y_PRED )
            }
    elif _Type == "regression":
        _Metrics = {
            "MAE" : metrics.mean_absolute_error( y_TEST, y_PRED ),
            "MSE" : metrics.mean_squared_error( y_TEST, y_PRED )
            }
    else:
        raise Exception()

    return y_PRED, _Metrics



def _PARA_GRIDDING( Model, X, y, param_grid, _Scale = False, _CVType = "KFold", n_splits = N_SPLITS, random_state = RANDOM_STATE, scoring = SCORING, test_size = TEST_SIZE ): # Fine!

    if _Scale:
        _Scaler = preprocessing.StandardScaler().fit(X = X)
        X = _Scaler.transform(X = X)

    if _CVType == "KFold":
        _Cross_Val = model_selection.KFold( n_splits = n_splits, random_state = random_state )
    elif _CVType == "LeaveOneOut":
        _Cross_Val = model_selection.LeaveOneOut()
    elif _CVType == "ShuffleSplit":
        _Cross_Val = model_selection.ShuffleSplit( n_splits = n_splits, test_size = test_size, random_state = random_state )
    else:
        raise Exception()

    _Grid = model_selection.GridSearchCV( estimator = Model, param_grid = param_grid, cv = _Cross_Val, scoring = scoring )
    _Grid_Result = _Grid.fit( X = X, y = y )

    return _Grid_Result



def _MODEL_SCALING(Models): # Fine!

    return { _Key : pipeline.Pipeline( steps = [ ( "Scaler", preprocessing.StandardScaler() ), ( _Key, Models[_Key] ) ] ) for _Key in Models }



def _GET_GRID_PARA(Model): # Fine!
    if ( type(Model) in SUPPORTED_MODELS_TO_GRID ) == False:
        raise Exception("")
    if type(Model) in [ KNeighborsClassifier, KNeighborsRegressor ]:
        return {
            "n_neighbors" : [ 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21 ]
            }
    elif type(Model) in [ SVC, SVR ]:
        return {
            "C" : [ 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1 ],
            "kernel" : [ "linear", "poly", "rbf", "sigmoid", "precomputed" ]
            }
    else:
        return None



def _USING_GRIDDED_PARA(Model, _Type = "classification", _Grid_Result = {}): # Fine!
    if _Type == "classification":
        if type(Model) == KNeighborsClassifier:
            return KNeighborsClassifier(n_neighbors = _Grid_Result.best_params_["n_neighbors"])
        elif type(Model) == SVC:
            return SVC( C = _Grid_Result.best_params_["C"], kernel = _Grid_Result.best_params_["kernel"] )
        else:
            raise Exception("")
    elif _Type == "regression":
        if type(Model) == KNeighborsRegressor:
            return KNeighborsRegressor(n_neighbors = _Grid_Result.best_params_["n_neighbors"])
        elif type(Model) == SVR:
            return SVR( kernel = _Grid_Result.best_params_["kernel"], C = _Grid_Result.best_params_["C"] )
        else:
            raise Exception
    else:
        raise Exception("")



def _HYPER_PREDICT( _Data, _Plot = False, _CVType = "KFold", n_splits = N_SPLITS, random_state = RANDOM_STATE, scoring = SCORING, test_size = TEST_SIZE, _Type = "classification" ):
    _X_Train, _X_Test, _y_Train, _y_Test = _AUTO_SPLITTING_DATASET( _Data, test_size = test_size, random_state = random_state )
    _Ordinary_Models = _GET_ORDINARY_MODEL_DICT(_Type = _Type)
    _Ensemble_Models = _GET_ORDINARY_MODEL_DICT(_Type = _Type)
    _Ordinary_Algorithms_Comparison_Results, _Best_Ordinary_Model = _ALGO_CMP( Models = _Ordinary_Models, X = _X_Train, y = _y_Train, _Plot = _Plot )
    _Ensemble_Algorithms_Comparison_Results, _Best_Ensemble_Model = _ALGO_CMP( Models = _Ensemble_Models, X = _X_Train, y = _y_Train, _Plot = _Plot )
    _y_Ordinary_Pred, _Metrics_Ordinary = _MODEL_RUN( Model = _Best_Ordinary_Model, X_TRAIN = _X_Train, X_TEST = _X_Test, y_TRAIN = _y_Train, y_TEST = _y_Test, _Type = _Type )
    _y_Ensemble_Pred, _Metrics_Ensemble = _MODEL_RUN( Model = _Best_Ensemble_Model, X_TRAIN = _X_Train, X_TEST = _X_Test, y_TRAIN = _y_Train, y_TEST = _y_Test, _Type = _Type )
    if _Type == "classification":
        return ( _y_Ordinary_Pred if ( _Metrics_Ordinary["ACCURACY_SCORE"] >= _Metrics_Ensemble["ACCURACY_SCORE"] ) else _y_Ensemble_Pred )
    elif _Type == "regression":
        return ( _y_Ordinary_Pred if ( ( _Metrics_Ordinary["MSE"] > _Metrics_Ensemble["MSE"] ) or ( ( _Metrics_Ordinary["MSE"] == _Metrics_Ensemble["MSE"] ) and ( _Metrics_Ordinary["MAE"] >= _Metrics_Ensemble["MAE"] ) ) ) else _y_Ensemble_Pred )
    else:
        raise Exception("")



# -*- END -*- #
