# Import Statements
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy


class Classification_ANN_Diagnoser:
    def __init__(self):
        self.data = np.loadtxt('data_w3_ex2.csv', delimiter=',')
        self.x, self.y = self.divide_x_y()
        self.x_train, self.y_train, \
        self.x_cv, self.y_cv, \
        self.x_test, self.y_test = self.split_data()
        self.x_train_scaled, self.x_cv_scaled, self.x_test_scaled = self.feature_scaling()
        self.threshold = 0.5
        self.models = self.build_models()
        self.nn_train_error = []
        self.nn_cv_error = []

    def divide_x_y(self):
        # Dividing features and targets
        x = self.data[:, :-1]
        y = self.data[:, 2:3]
        return x, y

    def split_data(self):
        # Splitting data into 3 sets
        x_train, x_, y_train, y_ = train_test_split(self.x, self.y, test_size=0.40, random_state=1)
        x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)
        # Delete x_, y_
        del x_, y_

        return x_train, y_train, x_cv, y_cv, x_test, y_test

    def feature_scaling(self):
        scaler_linear = StandardScaler()
        x_train_scaled = scaler_linear.fit_transform(self.x_train)
        x_cv_scaled = scaler_linear.transform(self.x_cv)
        x_test_scaled = scaler_linear.transform(self.x_test)
        return x_train_scaled, x_cv_scaled, x_test_scaled

    def plot_dataset(self, title):
        for i in range(len(self.y)):
            marker = 'x' if self.y[i] == 1 else 'o'
            c = 'r' if self.y[i] == 1 else 'b'
            plt.scatter(self.x[i, 0], self.x[i, 1], marker=marker, c=c)
        plt.title("x1 vs x2")
        plt.xlabel("x1")
        plt.ylabel("x2")
        y_0 = mlines.Line2D([], [], color='r', marker='x', markersize=12, linestyle='None', label='y=1')
        y_1 = mlines.Line2D([], [], color='b', marker='o', markersize=12, linestyle='None', label='y=0')
        plt.title(title)
        plt.legend(handles=[y_0, y_1])
        plt.show()

    @staticmethod
    def build_models():
        tf.random.set_seed(20)
        model_1 = Sequential(
            [
                Dense(25, activation='relu'),
                Dense(15, activation='relu'),
                Dense(1, activation='sigmoid')
            ],
            name='model_1'
        )

        model_2 = Sequential(
            [
                Dense(20, activation='relu'),
                Dense(12, activation='relu'),
                Dense(12, activation='relu'),
                Dense(20, activation='relu'),
                Dense(1, activation='sigmoid')
            ],
            name='model_2'
        )

        model_3 = Sequential(
            [
                Dense(32, activation='relu'),
                Dense(16, activation='relu'),
                Dense(8, activation='relu'),
                Dense(4, activation='relu'),
                Dense(12, activation='relu'),
                Dense(1, activation='sigmoid')
            ],
            name='model_3'
        )

        models = [model_1, model_2, model_3]

        return models

    def run_diagnostics(self):
        for model in self.models:
            model.compile(
                loss=BinaryCrossentropy(from_logits=True),
                optimizer=Adam(learning_rate=0.01),
            )

            print("Training {}...".format(model.name))

            model.fit(
                self.x_train_scaled, self.y_train,
                epochs=200,
                verbose=0
            )

            print("Done!\n")

            yhat = model.predict(self.x_train_scaled)
            yhat = tf.math.sigmoid(yhat)
            yhat = np.where(yhat >= self.threshold, 1, 0)
            train_error = np.mean(yhat != self.y_train)
            print(train_error)
            self.nn_train_error.append(train_error)

            yhat = model.predict(self.x_cv_scaled)
            yhat = tf.math.sigmoid(yhat)
            yhat = np.where(yhat >= self.threshold, 1, 0)
            cv_error = np.mean(yhat != self.y_cv)
            print(cv_error)
            self.nn_cv_error.append(cv_error)

        for model_num in range(len(self.nn_train_error)):
            print(
                "Model {}: Training Set Classification Error: {}\n".format(
                    model_num + 1, self.nn_train_error[model_num]
                )
            )
            print(
                "CV Set Classification Error: {}".format(
                    self.nn_cv_error[model_num]
                )
            )

            # Results: Train    Test
            # Model 1: 0.05833, 0.17500
            # Model 2: 0.06667, 0.15000
            # Model 3: 0.05000, 0.15000 <----Selected

    def final_testing(self, model_num):
        # Compute the test error
        yhat = self.models[model_num - 1].predict(self.x_test_scaled)
        yhat = tf.math.sigmoid(yhat)
        yhat = np.where(yhat >= self.threshold, 1, 0)
        nn_test_error = np.mean(yhat != self.y_test)

        print(f"Selected Model: {model_num}")
        print(f"Training Set Classification Error: {self.nn_train_error[model_num - 1]}")
        print(f"CV Set Classification Error: {self.nn_cv_error[model_num - 1]}")
        print(f"Test Set Classification Error: {nn_test_error}")

        # Results: Train    CV      Test
        # Model 3: 0.0500, 0.1500, 0.1750


diagnoser = Classification_ANN_Diagnoser()
diagnoser.plot_dataset("x1 Vs x2")
diagnoser.run_diagnostics()
# Select the model with the lowest error
diagnoser.final_testing(model_num=3)
