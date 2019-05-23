import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
import random
import time


data_train = pd.read_csv('data/poker-hand-training-true.data', header=None, sep=',')
data_test = pd.read_csv('data/poker-hand-testing.data', header=None, sep=',')
# print(data_train.shape, data_test.shape)

x_train, y_train = data_train.values[:, :-1], data_train.values[:, -1:]
x_test, y_test = data_test.values[:, :-1], data_test.values[:, -1:]
# print(x_train.shape, y_train.shape)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(x_test)
# scaler X = 0.536906
# StandardScaler() = 0.99409 / MinMaxScaler() = 0.545303 / MinMaxScaler() + relu = 0.837988

start_time = time.time()
clf = MLPClassifier(hidden_layer_sizes=(32, 32), solver='adam', activation='tanh',
                    alpha=0.01, learning_rate_init=0.01, max_iter=1000,
                    random_state=random.randint(0, 100))

result = clf.fit(x_train, y_train)
pred = clf.predict(x_test)
acc = accuracy_score(y_test, pred)
print(acc)
print(f'classifier time = {(time.time() - start_time)}')
# 'adam' => 'tanh' = 0.99409 / 'relu' = 0.718022 / 'logistic' = 0.546115 / 'identity' = 0.442899
# ‘lbfgs’ = 0.993829, ‘sgd’ = 0.98884, ‘adam’ = 0.995397
# 'lbfgs' = 53.12708377838135, 'sgd' = 63.72900986671448, 'adam' = 22.409477710723877

