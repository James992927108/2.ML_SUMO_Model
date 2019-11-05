from __future__ import absolute_import, division, print_function

import csv
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.contrib.keras.api.keras.wrappers.scikit_learn import \
    KerasRegressor

import Data_set

# import import_data
model_name_list = ["fitnesses", "complete_duration",
                   "waitingTime", "incomplete_Veh_num", "complete_Veh_num"]


def getData():
    data = Data_set.load_data()

    x = data.drop(model_name_list, axis=1)

    y = data.drop(x, axis=1)
    print(list(y))

    y["complete_duration"] = pd.to_numeric(y["complete_duration"])
    y["waitingTime"] = pd.to_numeric(y["waitingTime"])
    y["incomplete_Veh_num"] = pd.to_numeric(y["incomplete_Veh_num"])
    y["complete_Veh_num"] = pd.to_numeric(y["complete_Veh_num"])

    model_train_test_list = []
    for model_name in model_name_list:
        print(data[model_name].head(1))
        x_train, x_test, y_train, y_test = train_test_split(
            x, data[model_name], random_state=40)
        model_train_test_list.append([x_train, x_test, y_train, y_test])

    return model_train_test_list


def sklearn_linear_regressor():
    np.set_printoptions(precision=30, suppress=True)

    model_train_test_list = getData()

    # x_train, x_test, y_train, y_test
    linreg = LinearRegression()
    model = linreg.fit(model_train_test_list[0][0], model_train_test_list[0][2])

    # print(linreg.coef_)

    # # save the model to disk
    # filename = 'linear_version_model.sav'
    # pickle.dump(model, open(filename, 'wb'))
    # # load the model
    # loaded_model = pickle.load(open(filename, 'rb'))
    # y_pred = loaded_model.predict(x_test)

    # print y_test[-100:]
    # print y_pred[-100:]
    # result = loaded_model.score(x_test, y_test)
    # print result

    # print model
    # print linreg.intercept_

    # y_pred = linreg.predict(x_test)
    # print y_pred

    # plt.figure()
    # plt.plot(range(100),y_pred[-100:],'b',label="fitness from model")
    # plt.plot(range(100),y_test[-100:],'r',label="fitness from sumo")
    # plt.legend(loc="upper right")
    # plt.xlabel("parmeter")
    # plt.ylabel("fitnesses")
    # plt.show()


def keras_linear_regression():
    model_train_test_list = getData()

    # for i in range(1,len(model_train_test_list)):
    i = 4
    print(model_name_list[i])
    model = Sequential()
    model.add(Dense(units=512, input_dim=model_train_test_list[i][0].shape[1]))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1))

    Adam = optimizers.Adam(lr=0.001, decay=0.004)

    model.compile(loss='mse', optimizer=Adam)

    print('Training -----------')
    cost = model.fit(
        model_train_test_list[i][0], model_train_test_list[i][2], batch_size=128, epochs=100)
    model_name =  'model/{}.h5'.format(str(model_name_list[i]))
    model.save(model_name)

    print('\nTesting ------------')
    cost = model.evaluate(
        model_train_test_list[i][1], model_train_test_list[i][3], batch_size=128)
    print('test cost:', cost)
    # # W, b = model.layers[0].get_weights()
    # # print('Weights=', W, '\nbiases=', b)
    # y_pred_1 = model.predict(x_test)

    # loaded_model = tf.contrib.keras.models.load_model(model_name)
    # y_pred = loaded_model.predict(x_test)

def tf_linear_version(argv):
    STEPS = 1000
    PRICE_NORM_FACTOR = 100
    assert len(argv) == 1
    (train, eval) = import_data.dataset()

    def to_thousands(features, labels):
        return features, labels / PRICE_NORM_FACTOR

    train = train.map(to_thousands)
    eval = eval.map(to_thousands)

    def input_train():
        return (train.shuffle(1000).batch(128).repeat().make_one_shot_iterator().get_next())

    def input_eval():
        return (eval.shuffle(1000).batch(128).repeat().make_one_shot_iterator().get_next())

    feature_columns = [
        tf.feature_column.numeric_column(key="c0"),
        tf.feature_column.numeric_column(key="c1"),
        tf.feature_column.numeric_column(key="c2"),
        tf.feature_column.numeric_column(key="c3"),
        tf.feature_column.numeric_column(key="c4"),
        tf.feature_column.numeric_column(key="c5"),
        tf.feature_column.numeric_column(key="c6"),
        tf.feature_column.numeric_column(key="c7"),
        tf.feature_column.numeric_column(key="c8"),
        tf.feature_column.numeric_column(key="c9"),
        tf.feature_column.numeric_column(key="c10")
    ]
    # Build the Estimator.
    DNN = False
    if(DNN == True):
        model = tf.estimator.DNNRegressor(
            hidden_units=[100, 50, 10, 5], feature_columns=feature_columns, optimizer='Ftrl')
    else:
        model = tf.estimator.LinearRegressor(feature_columns=feature_columns)
        # 'Adagrad', 'Adam', 'Ftrl'

    model.train(input_fn=input_train, steps=STEPS)

    eval_result = model.evaluate(input_fn=input_eval, steps=STEPS)

    average_loss = eval_result["average_loss"]
    print("\n" + 50 * "*")
    print("\nRMS error for the test set: ${}"
          .format(PRICE_NORM_FACTOR * average_loss**0.5))
    # ---------------------------------------------------------------
    x_train, y_train, x_test = getData()
    input_dict = {
        "c0": np.array(x_test.c0),
        "c1": np.array(x_test.c1),
        "c2": np.array(x_test.c2),
        "c3": np.array(x_test.c3),
        "c4": np.array(x_test.c4),
        "c5": np.array(x_test.c5),
        "c6": np.array(x_test.c6),
        "c7": np.array(x_test.c7),
        "c8": np.array(x_test.c8),
        "c9": np.array(x_test.c9),
        "c10": np.array(x_test.c10),
    }
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        input_dict, shuffle=False)
    predict_results = model.predict(input_fn=predict_input_fn)
    # predict_results.
    print("\nPrediction results:")
    for i, prediction in enumerate(predict_results):
        msg = ("{} {} {}".format
               (
                   i,
                   PRICE_NORM_FACTOR * prediction["predictions"][0],
                   np.array(y_test[i:i+1]).flatten()
               )
               )
        print("    " + msg)


if __name__ == "__main__":
    # sklearn_linear_regressor()
    keras_linear_regression()
    # tf.logging.set_verbosity(tf.logging.INFO)
    # tf.app.run(main=tf_linear_version)
