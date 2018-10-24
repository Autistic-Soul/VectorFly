#!/usr/bin/env Python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use("seaborn")

from sklearn import model_selection, \
                    preprocessing, \
                    pipeline, \
                    metrics
from sklearn.externals import joblib

from sklearn.linear_model import LinearRegression, \
                                 Ridge, \
                                 Lasso, \
                                 ElasticNet, \
                                 LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, \
                              KNeighborsRegressor
from sklearn.svm import SVC, \
                        SVR
from sklearn.tree import ExtraTreeClassifier, \
                         ExtraTreeRegressor, \
                         DecisionTreeClassifier, \
                         DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesClassifier, \
                             ExtraTreesRegressor, \
                             RandomForestClassifier, \
                             RandomForestRegressor, \
                             AdaBoostClassifier, \
                             AdaBoostRegressor, \
                             GradientBoostingClassifier, \
                             GradientBoostingRegressor

N_SPLITS = 10
RANDOM_STATE = 7
TEST_SIZE = 0.2
SCORING = "accuracy"
SHUFFLE = False

MODELS_CAN_BE_GRIDDED = [
    KNeighborsClassifier,
    KNeighborsRegressor,
    SVC,
    SVR,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
    ]

def GET_ORDINARY_MODELS(prediction_type = None):
    if prediction_type == "C":
        return {
            "LR" : LogisticRegression(),
            "LDA" : LinearDiscriminantAnalysis(),
            "GNB" : GaussianNB(),
            "KNC" : KNeighborsClassifier(),
            "SVC" : SVC(),
            "ETC" : ExtraTreeClassifier(),
            "DTC" : DecisionTreeClassifier()
            }
    elif prediction_type == "R":
        return {
            "LR" : LinearRegression(),
            "RIDGE" : Ridge(),
            "LASSO" : Lasso(),
            "EN" : ElasticNet(),
            "KNR" : KNeighborsRegressor(),
            "SVR" : SVR(),
            "ETR" : ExtraTreeRegressor(),
            "DTR" : DecisionTreeRegressor()
            }
    else:
        raise Exception()

def GET_ENSEMBLE_MODELS(prediction_type = None):
    if prediction_type == "C":
        return {
            "ETC_Ensemble" : ExtraTreesClassifier(),
            "RFC_Ensemble" : RandomForestClassifier(),
            "ABC_Ensemble" : AdaBoostClassifier(),
            "GBC_Ensemble" : GradientBoostingClassifier()
            }
    elif prediction_type == "R":
        return {
            "ETR_Ensemble" : ExtraTreesRegressor(),
            "RFR_Ensemble" : RandomForestRegressor(),
            "ABR_Ensemble" : AdaBoostRegressor(),
            "GBR_Ensemble" : GradientBoostingRegressor()
        }
    else:
        raise Exception()

def GET_ALLKINDS_MODELS(prediction_type = None):
    if prediction_type == "C":
        return {
            "LR" : LogisticRegression(),
            "LDA" : LinearDiscriminantAnalysis(),
            "GNB" : GaussianNB(),
            "KNC" : KNeighborsClassifier(),
            "SVC" : SVC(),
            "ETC" : ExtraTreeClassifier(),
            "DTC" : DecisionTreeClassifier(),
            "ETC_Ensemble" : ExtraTreesClassifier(),
            "RFC_Ensemble" : RandomForestClassifier(),
            "ABC_Ensemble" : AdaBoostClassifier(),
            "GBC_Ensemble" : GradientBoostingClassifier()
            }
    elif prediction_type == "R":
        return {
            "LR" : LinearRegression(),
            "RIDGE" : Ridge(),
            "LASSO" : Lasso(),
            "EN" : ElasticNet(),
            "KNR" : KNeighborsRegressor(),
            "SVR" : SVR(),
            "ETR" : ExtraTreeRegressor(),
            "DTR" : DecisionTreeRegressor(),
            "ETR_Ensemble" : ExtraTreesRegressor(),
            "RFR_Ensemble" : RandomForestRegressor(),
            "ABR_Ensemble" : AdaBoostRegressor(),
            "GBR_Ensemble" : GradientBoostingRegressor()
            }
    else:
        raise Exception()

def SPLIT_DATA( data_tobe_splitted, test_size = TEST_SIZE, random_state = RANDOM_STATE ):
    return model_selection.train_test_split(
        data_tobe_splitted.values[ : , : (-1) ],
        data_tobe_splitted.values[ : , (-1) ],
        test_size = test_size,
        random_state = random_state
        )

def NEW_SCALER(scaler_type = "StandardScaler"):
    if scaler_type == "StandardScaler":
        return preprocessing.StandardScaler()
    elif scaler_type == "Binarizer":
        return preprocessing.Binarizer()
    elif scaler_type == "Normalizer":
        return preprocessing.Normalizer()
    else:
        raise Exception()

def MODEL_SCALING( models, scaler_type = "StandardScaler" ):
    scaler = NEW_SCALER(scaler_type = scaler_type)
    return { key : pipeline.Pipeline([ ("Scaler", scaler), (key, models[key]) ]) for key in models }

def MODEL_CV( cv_type = "KFold", n_splits = N_SPLITS, random_state = RANDOM_STATE, test_size = TEST_SIZE, scoring = SCORING, shuffle = SHUFFLE ):
    if cv_type == "KFold":
        return model_selection.KFold( n_splits = n_splits, shuffle = shuffle, random_state = random_state )
    elif cv_type == "LeaveOneOut":
        return model_selection.LeaveOneOut()
    elif cv_type == "ShuffleSplit":
        return model_selection.ShuffleSplit( n_splits = n_splits, test_size = test_size, random_state = random_state )
    else:
        raise Exception()

def ALGO_CMP( models = None, X_TRAIN = None, y_TRAIN = None, prediction_type = None, scale = False, scaler_type = "StandardScaler", cv_type = "KFold", n_splits = N_SPLITS, random_state = RANDOM_STATE, test_size = TEST_SIZE, scoring = SCORING, shuffle = SHUFFLE ):

    if models == None:
        models = GET_ALLKINDS_MODELS(prediction_type = "C")

    if scale:
        NEW_SCALER(scaler_type = scaler_type).fit_transform(X = X_TRAIN)

    cross_validator = MODEL_CV( cv_type = cv_type, n_splits = n_splits, random_state = random_state, test_size = test_size, scoring = scoring, shuffle = shuffle )

    results = [ ( models[each_model], model_selection.cross_val_score( estimator = models[each_model], X = X_TRAIN, y = y_TRAIN, scoring = scoring, cv = cross_validator ) ) for each_model in models ]

    for each_result in results:
        print( "Algorithm: %s,\nMEAN: %f, STD: %f" % ( type(each_result[0]), each_result[1].mean(), each_result[1].std() ) )
    plt.title("Algorithm_Comparison")
    plt.boxplot( x = [ each_result[1] for each_result in results ], labels = models.keys() )
    plt.show()
    print()

    best_model = results[0]
    for i in range( 1, len(results) ):
        if results[i][1].mean() > best_model[1].mean():
            best_model = results[i]

    return best_model[0], [ each[1] for each in results ]

class ALGORITHM_COMPARISON(object):

    def __init__( self, models = None, X_TRAIN = None, y_TRAIN = None, prediction_type = None, scale = False, scaler_type = "StandardScaler", cv_type = "KFold", n_splits = N_SPLITS, random_state = RANDOM_STATE, test_size = TEST_SIZE, scoring = SCORING, shuffle = SHUFFLE ):

        if models == None:
            models = GET_ALLKINDS_MODELS(prediction_type = prediction_type)

        if scale:
            NEW_SCALER(scaler_type = scaler_type).fit_transform(X = X_TRAIN)

        cross_validator = MODEL_CV( cv_type = cv_type, n_splits = n_splits, random_state = random_state, test_size = test_size, scoring = scoring, shuffle = shuffle )

        results = [ ( models[each_model], model_selection.cross_val_score( estimator = models[each_model], X = X_TRAIN, y = y_TRAIN, scoring = scoring, cv = cross_validator ) ) for each_model in models ]

        best = results[0]
        for i in range( 1, len(results) ):
            if results[i][1].mean() > best[1].mean():
                best = results[i]

        self.__best_model = best[0]
        self.__best_result = best[1]
        self.__all_keys = models.keys()
        self.__all_results = results

        return

    def PRINT_RESULTS(self):
        for each_result in self.__all_results:
            model_type_name = type(each_result[0])
            if model_type_name == pipeline.Pipeline:
                print( "Pipelined", end = " " )
                model_type_name = type(each_result[0].steps[-1][-1])
            print( "Algorithm: %s" % model_type_name )
            print( "MEAN: %f, STD: %f" % ( each_result[1].mean(), each_result[1].std() ) )
        print()
        return

    def PLOT_FIGURE( self, figure_title = "Algorithm_Comparison" ):
        plt.title(s = figure_title)
        plt.boxplot( x = [ each_result[1] for each_result in self.__all_results ], labels = self.__all_keys )
        plt.show()
        return

    def BEST_MODEL(self):
        return self.__best_model

    def BEST_RESULT(self):
        return self.__best_result

def MODEL_PREDICT( model = None, X_TRAIN = None, X_TEST = None, y_TRAIN = None, y_TEST = None, prediction_type = None, scale = False, scaler_type = "StandardScaler" ):

    if scale:
        scaler = NEW_SCALER(scaler_type = scaler_type).fit_transform(X = X_TRAIN)
        X_TEST = scaler.transform(X = X_TEST)

    model.fit( X = X_TRAIN, y = y_TRAIN )
    y_PRED = model.predict(X = X_TEST)

    if prediction_type == "C":
        prediction_result = {
            "ACCURACY_SCORE" : metrics.accuracy_score( y_true = y_TEST, y_pred = y_PRED ),
            "CONFUSION_MATRIX" : metrics.confusion_matrix( y_true = y_TEST, y_pred = y_PRED ),
            "CLASSIFICATION_REPORT" : metrics.classification_report( y_true = y_TEST, y_pred = y_PRED )
            }
    elif prediction_type == "R":
        prediction_result = {
            "L1-MAE" : metrics.mean_absolute_error( y_true = y_TEST, y_pred = y_PRED ),
            "L2-MSE" : metrics.mean_squared_error( y_true = y_TEST, y_pred = y_PRED )
            }
    else:
        raise Exception()

    return y_PRED, prediction_result

class metrics_results(object):
    def __init__( self, prediction_type = None, y_TRUE = None, y_PRED = None ):
        if prediction_type == "C":
            self.accuracy_score = metrics.accuracy_score( y_true = y_TRUE, y_pred = y_PRED )
            self.confusion_matrix = metrics.confusion_matrix( y_true = y_TRUE, y_pred = y_PRED )
            self.classification_report = metrics.classification_report( y_true = y_TRUE, y_pred = y_PRED )
        elif prediction_type == "R":
            self.L1_MAE = metrics.mean_absolute_error( y_true = y_TRUE, y_pred = y_PRED )
            self.L2_MSE = metrics.mean_squared_error( y_true = y_TRUE, y_pred = y_PRED )
        else:
            raise Exception()
        return

class MODEL_PREDICTION(object):

    def __init__( self, model = None, X_TRAIN = None, X_TEST = None, y_TRAIN = None, y_TEST = None, prediction_type = None, scale = False, scaler_type = "StandardScaler" ):

        if scale:
            scaler = NEW_SCALER(scaler_type = scaler_type).fit_transform(X = X_TRAIN)
            X_TEST = scaler.transform(X = X_TEST)

        model.fit( X = X_TRAIN, y = y_TRAIN )

        self.__y_PRED = model.predict(X = X_TEST)
        self.__prediction_report = metrics_results( prediction_type = prediction_type, y_TRUE = y_TEST, y_PRED = self.__y_PRED )

        return

    def y_PRED(self):
        return self.__y_PRED

    def REPORT(self):
        return self.__prediction_report

def GET_GRID_PARA(model = None):
    if type(model) == pipeline.Pipeline:
        return GET_GRID_PARA(model = model.steps[-1][-1])
    elif type(model) in [ KNeighborsClassifier, KNeighborsRegressor ]:
        return { "n_neighbors" : [ 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21 ] }
    elif type(model) in [ SVC, SVR ]:
        return { "C" : [ 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9 ], "kernel" : [ "linear", "poly", "rbf", "sigmoid" ] }
    elif type(model) in [ AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor ]:
        return { "n_estimators" : [ 10, 20, 50, 100 ] }
    elif type(model) in [ DecisionTreeClassifier, ExtraTreeClassifier ]:
        return { "criterion" : [ "gini", "entropy" ], "splitter" : [ "best", "random" ] }
    elif type(model) in [ DecisionTreeRegressor, ExtraTreeRegressor ]:
        return { "criterion" : [ "mse", "mae" ], "splitter" : [ "best", "random" ] }
    else:
        return None

def BEST_PARA_SEARCH( model = None, param_grid = None, X_TRAIN = None, y_TRAIN = None, scale = False, scaler_type = "StandardScaler", cv_type = "KFold", n_splits = N_SPLITS, random_state = RANDOM_STATE, test_size = TEST_SIZE, scoring = SCORING, shuffle = SHUFFLE ):

    model_type_name = type(model)
    if model_type_name == pipeline.Pipeline:
        print( "Pipelined", end = " " )
        model_type_name = type(model.steps[-1][-1])
        model = model.steps[-1][-1]

    if param_grid == None:
        param_grid = GET_GRID_PARA(model = model)

    if scale:
        NEW_SCALER(scaler_type = scaler_type).fit_transform(X = X_TRAIN)
    cross_validator = MODEL_CV( cv_type = cv_type, n_splits = n_splits, random_state = random_state, test_size = test_size, scoring = scoring, shuffle = shuffle )
    grid_cv = model_selection.GridSearchCV( estimator = model, param_grid = param_grid, scoring = scoring, cv = cross_validator )

    result = grid_cv.fit( X = X_TRAIN, y = y_TRAIN )

    cv_results = zip( result.cv_results_["mean_test_score"], \
                      result.cv_results_["std_test_score"], \
                      result.cv_results_["params"]
                      )

    print( "Algorithm: %s" % model_type_name )
    print( "Best Score: %s" % result.best_score_ )
    print( "Best Parameter: %s" % result.best_params_ )
    for each_mean_test_score, each_std_test_score, each_params in cv_results:
        print( "%f (%f) with %r" % ( each_mean_test_score, each_std_test_score, each_params ) )
    print()

    return result.best_params_

def MODEL_WITH_BESTPARA( model = None, param = None ):
    if type(model) == pipeline.Pipeline:
        returning_model = MODEL_WITH_BESTPARA( model = model.steps[-1][-1], param = param )
        stp = model.steps
        stp[-1] = list(stp[-1])
        stp[-1][-1] = returning_model
        stp[-1] = tuple(stp[-1])
        return pipeline.Pipeline(steps = stp)
    elif type(model) == KNeighborsClassifier:
        return KNeighborsClassifier(n_neighbors = param["n_neighbors"])
    elif type(model) == KNeighborsRegressor:
        return KNeighborsRegressor(n_neighbors = param["n_neighbors"])
    elif type(model) == SVC:
        return SVC( C = param["C"], kernel = param["kernel"] )
    elif type(model) == SVR:
        return SVR( C = param["C"], kernel = param["kernel"] )
    elif type(model) == AdaBoostClassifier:
        return AdaBoostClassifier(n_estimators = param["n_estimators"])
    elif type(model) == AdaBoostRegressor:
        return AdaBoostRegressor(n_estimators = param["n_estimators"])
    elif type(model) == GradientBoostingClassifier:
        return GradientBoostingClassifier(n_estimators = param["n_estimators"])
    elif type(model) == GradientBoostingRegressor:
        return GradientBoostingRegressor(n_estimators = param["n_estimators"])
    elif type(model) == ExtraTreesClassifier:
        return ExtraTreesClassifier(n_estimators = param["n_estimators"])
    elif type(model) == ExtraTreesRegressor:
        return ExtraTreesRegressor(n_estimators = param["n_estimators"])
    elif type(model) == RandomForestClassifier:
        return RandomForestClassifier(n_estimators = param["n_estimators"])
    elif type(model) == RandomForestRegressor:
        return RandomForestRegressor(n_estimators = param["n_estimators"])
    elif type(model) == DecisionTreeClassifier:
        return DecisionTreeClassifier( criterion = param["criterion"], splitter = param["splitter"] )
    elif type(model) == DecisionTreeRegressor:
        return DecisionTreeRegressor( criterion = param["criterion"], splitter = param["splitter"] )
    elif type(model) == ExtraTreeClassifier:
        return ExtraTreeClassifier( criterion = param["criterion"], splitter = param["splitter"] )
    elif type(model) == ExtraTreeRegressor:
        return ExtraTreeRegressor( criterion = param["criterion"], splitter = param["splitter"] )
    else:
        return model

class HYPER_PREDICTION(object):

    def __init__( self, prediction_type = None, X_TRAIN = None, X_TEST = None, y_TRAIN = None, y_TEST = None, models = None, grid_parameter_fit = True, n_splits = N_SPLITS, random_state = RANDOM_STATE, test_size = TEST_SIZE, scoring = SCORING, shuffle = SHUFFLE, cv_type = "KFold", scale = False, scaler_type = "StandardScaler" ):

        if models == None:
            models = GET_ALLKINDS_MODELS(prediction_type = prediction_type)

        if scale:
            models = MODEL_SCALING( models, scaler_type = scaler_type )
            scaler = NEW_SCALER(scaler_type = scaler_type).fit(X = X_TRAIN)
            X_TRAIN = scaler.transform(X = X_TRAIN)
            X_TEST = scaler.transform(X = X_TEST)

        self.__X_TRAIN = X_TRAIN
        self.__X_TEST = X_TEST
        self.__y_TRAIN = y_TRAIN
        self.__y_TEST = y_TEST

        self.__prediction_type = prediction_type

        if grid_parameter_fit:
            for key in models:
                if ( type(models[key]) in MODELS_CAN_BE_GRIDDED ) or ( ( type(models[key]) == pipeline.Pipeline ) and ( type(models[key].steps[-1][-1]) in MODELS_CAN_BE_GRIDDED ) ):
                    models[key] = MODEL_WITH_BESTPARA( model = models[key], param = BEST_PARA_SEARCH( model = models[key], param_grid = GET_GRID_PARA(model = models[key]), X_TRAIN = X_TRAIN, y_TRAIN = y_TRAIN, cv_type = cv_type, n_splits = n_splits, random_state = random_state, test_size = test_size, scoring = scoring, shuffle = shuffle ) )

        self.__models = models

        self.__models_comparison = ALGORITHM_COMPARISON( models = self.__models, X_TRAIN = self.__X_TRAIN, y_TRAIN = self.__y_TRAIN, prediction_type = prediction_type, cv_type = cv_type, n_splits = n_splits, random_state = random_state, test_size = test_size, scoring = scoring, shuffle = shuffle )
        self.__models_comparison.PRINT_RESULTS()
        self.__models_comparison.PLOT_FIGURE()

        self.__Algorithm = self.__models_comparison.BEST_MODEL()

        self.__Prediction = MODEL_PREDICTION( model = self.__Algorithm, X_TRAIN = self.__X_TRAIN, X_TEST = self.__X_TEST, y_TRAIN = self.__y_TRAIN, y_TEST = self.__y_TEST, prediction_type = prediction_type )

        self.__y_PRED = self.__Prediction.y_PRED()
        self.__Prediction_Report = self.__Prediction.REPORT()
        self.__Prediction_Score = self.__models_comparison.BEST_RESULT()

        if type(self.__Algorithm) == pipeline.Pipeline:
            print( "Best (Pipelined) Algorithm:", self.__Algorithm.steps[-1][-1], sep = "\n" )
        else:
            print( "Best Algorithm:", self.__Algorithm, sep = "\n" )
        print( " " * 3, "MEAN:", self.__Prediction_Score.mean() )
        print( " " * 3, "STD:", self.__Prediction_Score.std() )

        return

    def ALGORITHM(self):
        return self.__Algorithm

    def y_PRED(self):
        return self.__y_PRED

    def REPORT(self):
        return self.__Prediction_Report

    def SCORE(self):
        return self.__Prediction_Score

    def PRINT_REPORT(self):
        if self.__prediction_type == "C":
            print("CLASSIFICATION_REPORT:")
            print(self.__Prediction_Report.classification_report)
            print()
            print("ACCURACY_SCORE:", end = " ")
            print(self.__Prediction_Report.accuracy_score)
            print()
            print("CONFUSION_MATRIX:")
            print(self.__Prediction_Report.confusion_matrix)
            print()
        elif self.__prediction_type == "R":
            print("MAE:", self.__Prediction_Report.L1_MAE)
            print("MSE:", self.__Prediction_Report.L2_MSE)
            print()

    def PREDICT( self, X ):
        return self.__Algorithm.predict(X)

    def DUMP_MODEL( self, filename = None ):
        joblib.dump( value = self.__Algorithm, filename = filename )
        return

    def ALL_MODELS(self):
        return self.__models

class DATA_CLEANING(object):

    """
    csv_path:
    ===
        <str>
    names:
    ===
        <list>
    losts:
    ===
        <list>
    fill:
    ===
        <str>: "MEAN" or "MEDIAN" or "MODE"
    """

    def __init__( self, csv_path, names = None, losts = None, fill_type = "MEAN", test_size = TEST_SIZE, random_state = RANDOM_STATE ):

        dataset = pd.read_csv(csv_path) if names == None else pd.read_csv(csv_path, names = names)
        self.__data = dataset
        self.__data = list(self.__data.values)

        for i in range(len(self.__data[0])):

            array = []
            for j in range(len(self.__data)):
                if not self.__data[j][i] in losts:
                    array.append(self.__data[j][i])
            
            if fill_type == "MEAN":
                sigma = 0
                for i in range(len(array)):
                    if not array[i] in losts:
                        sigma += array[i]
                filling = sigma / len(array)
            elif fill_type == "MEDIAN":
                array = array.sort()
                filling = array[ len(array) // 2 ]
            elif fill_type == "MODE":
                array = array.sort()
                filling = array[ len(array) // 2 ]
            else:
                raise Exception()

            for j in range(len(self.__data)):
                if self.__data[j][i] in losts:
                    self.__data[j][i] = filling

        col = dataset.columns
        self.__data = pd.DataFrame(self.__data, columns = col)

        self.X_Train, self.X_Test, self.y_Train, self.y_Test = SPLIT_DATA(self.__data, test_size, random_state)

    def CLEANED_DATA(self):
        return self.X_Train, self.X_Test, self.y_Train, self.y_Test





































iris_names = [ "sepal-length", "sepal-width", "petal-length", "petal-width" ]
sonar_names = [ 'SONAR1', 'SONAR2', 'SONAR3', 'SONAR4', 'SONAR5', 'SONAR6', 'SONAR7', 'SONAR8', 'SONAR9', 'SONAR10', 'SONAR11', 'SONAR12', 'SONAR13', 'SONAR14', 'SONAR15', 'SONAR16', 'SONAR17', 'SONAR18', 'SONAR19', 'SONAR20', 'SONAR21', 'SONAR22', 'SONAR23', 'SONAR24', 'SONAR25', 'SONAR26', 'SONAR27', 'SONAR28', 'SONAR29', 'SONAR30', 'SONAR31', 'SONAR32', 'SONAR33', 'SONAR34', 'SONAR35', 'SONAR36', 'SONAR37', 'SONAR38', 'SONAR39', 'SONAR40', 'SONAR41', 'SONAR42', 'SONAR43', 'SONAR44', 'SONAR45', 'SONAR46', 'SONAR47', 'SONAR48', 'SONAR49', 'SONAR50', 'SONAR51', 'SONAR52', 'SONAR53', 'SONAR54', 'SONAR55', 'SONAR56', 'SONAR57', 'SONAR58', 'SONAR59', 'SONAR60', 'class' ]
poker_names = [ "S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "CLASS" ]
ttt_names = [ "top-left-square", "top-middle-square", "top-right-square", "middle-left-square", "middle-middle-square", "middle-right-square", "bottom-left-square", "bottom-middle-square", "bottom-right-square", "class" ]

"""
iris = pd.read_csv( "C:\\iris.csv", names = iris_names )
XTrain, XTest, yTrain, yTest = SPLIT_DATA(iris)
iris_prediction = HYPER_PREDICTION(
    prediction_type = "C",
    X_TRAIN = XTrain,
    X_TEST = XTest,
    y_TRAIN = yTrain,
    y_TEST = yTest,
    scale = False
    )
"""

"""
sonar = pd.read_csv( "C:\\sonar.all-data.csv", names = sonar_names )
XTrain, XTest, yTrain, yTest = SPLIT_DATA(sonar)
sonar_prediction = HYPER_PREDICTION(
    prediction_type = "C",
    X_TRAIN = XTrain,
    X_TEST = XTest,
    y_TRAIN = yTrain,
    y_TEST = yTest,
    scale = False
    )
"""

"""
Poker_Train = pd.read_csv( "C:\\poker-hand-testing.csv", names = poker_names )
Poker_Test = pd.read_csv( "C:\\poker-hand-training-true.csv", names = poker_names )
XTrain = Poker_Train.values[ : , : (-1) ]
yTrain = Poker_Train.values[ : , (-1) ]
XTest = Poker_Test.values[ : , : (-1) ]
yTest = Poker_Test.values[ : , (-1) ]
poker_prediction = HYPER_PREDICTION(
    prediction_type = "C",
    X_TRAIN = XTrain,
    X_TEST = XTest,
    y_TRAIN = yTrain,
    y_TEST = yTest,
    grid_parameter_fit = False,
    scale = False
    )
"""

ttt = pd.read_csv( "C:\\tic-tac-toe.csv", names = ttt_names )
XTrain, XTest, yTrain, yTest = SPLIT_DATA(ttt)
ttt_prediction = HYPER_PREDICTION(
    prediction_type = "C",
    X_TRAIN = XTrain,
    X_TEST = XTest,
    y_TRAIN = yTrain,
    y_TEST = yTest,
    grid_parameter_fit = False,
    scale = False
    )

input()
