# 특이점
# 1. 학습 데이터(25,010)가 시험 데이터(1,000,000)의 2.5%에 불과
# 2. 무늬와 숫자가 한 쌍, 5개의 쌍(5장의 카드)와 족보로 구성 (피처 11개)

# CLASS
# 0: Nothing in hand; not a recognized poker hand
# 1: One pair; one pair of equal ranks within five cards
# 2: Two pairs; two pairs of equal ranks within five cards
# 3: Three of a kind; three equal ranks within five cards
# 4: Straight; five cards, sequentially ranked with no gaps
# 5: Flush; five cards with the same suit
# 6: Full house; pair + different rank three of a kind
# 7: Four of a kind; four equal ranks within five cards
# 8: Straight flush; straight + flush
# 9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
import collections
from imblearn import over_sampling


def sort_poker(x):
    # tuples = [[(i[0], i[1]), (i[2], i[3]), (i[4], i[5]), (i[6], i[7]), (i[8], i[9])] for i in x]
    tuples = [[(i[j], i[j+1]) for j in range(0, len(i), 2)] for i in x]
    # print(*tuples[:7], sep='\n')
    # [(1, 10), (1, 11), (1, 13), (1, 12), (1, 1)]
    # [(2, 11), (2, 13), (2, 10), (2, 12), (2, 1)]
    # [(3, 12), (3, 11), (3, 13), (3, 10), (3, 1)]
    # [(4, 10), (4, 11), (4, 1), (4, 13), (4, 12)]
    # [(4, 1), (4, 13), (4, 12), (4, 11), (4, 10)]
    # [(1, 2), (1, 4), (1, 5), (1, 3), (1, 6)]
    # [(1, 9), (1, 12), (1, 10), (1, 11), (1, 13)]

    tuples = [sorted(i, key=lambda t: t[0] * 100 + t[1]) for i in tuples]
    # print(*tuples[:7], sep='\n')
    # [(1, 1), (1, 10), (1, 11), (1, 12), (1, 13)]
    # [(2, 1), (2, 10), (2, 11), (2, 12), (2, 13)]
    # [(3, 1), (3, 10), (3, 11), (3, 12), (3, 13)]
    # [(4, 1), (4, 10), (4, 11), (4, 12), (4, 13)]
    # [(4, 1), (4, 10), (4, 11), (4, 12), (4, 13)]
    # [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]
    # [(1, 9), (1, 10), (1, 11), (1, 12), (1, 13)]

    x = [[k for j in i for k in j] for i in tuples]
    print(*x[:7], sep='\n')
    # [1, 1, 1, 10, 1, 11, 1, 12, 1, 13]
    # [2, 1, 2, 10, 2, 11, 2, 12, 2, 13]
    # [3, 1, 3, 10, 3, 11, 3, 12, 3, 13]
    # [4, 1, 4, 10, 4, 11, 4, 12, 4, 13]
    # [4, 1, 4, 10, 4, 11, 4, 12, 4, 13]
    # [1, 2, 1, 3, 1, 4, 1, 5, 1, 6]
    # [1, 9, 1, 10, 1, 11, 1, 12, 1, 13]

    return np.int32(x)


def load_poker_hands(file_path, training):
    names = ['S1', 'R1', 'S2', 'R2', 'S3', 'R3', 'S4', 'R4', 'S5', 'R5', 'CLASS']
    hands = pd.read_csv(file_path, sep=',', header=None, names=names)

    # print(hands.head())
    #    S1  R1  S2  R2  S3  R3  S4  R4  S5  R5  CLASS
    # 0   1  10   1  11   1  13   1  12   1   1      9
    # 1   2  11   2  13   2  10   2  12   2   1      9
    # 2   3  12   3  11   3  13   3  10   3   1      9
    # 3   4  10   4  11   4   1   4  13   4  12      9
    # 4   4   1   4  13   4  12   4  11   4  10      9

    y = hands['CLASS'].values
    x = hands.drop(['CLASS'], axis=1).values

    if training:
        x = sort_poker(x)

    x = preprocessing.OneHotEncoder(sparse=False).fit_transform(x)
    x = np.int32(x)

    return x, y


def build_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])
    return model


def show_accuracy(x_train, y_train, x_test, y_test):
    model = build_model([x_train.shape[1]])

    # EarlyStopping 기본은 val_loss.
    callbacks = tf.keras.callbacks.EarlyStopping(patience=5)
    model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=[callbacks],
              validation_split=0.2)

    print('acc :', model.evaluate(x_test, y_test))


x_train, y_train = load_poker_hands('poker/poker-hand-training-true.data', True)    # (25010, 11)
# x_test , y_test  = load_poker_hands('poker/poker-hand-testing.data', True)         # (1000000, 11)
#
# show_accuracy(x_train, y_train, x_test, y_test)
# acc : [0.2178394948476255, 0.959848]

# test set 정렬하지 않았을 때
# Epoch 82/100
# 20008/20008 [=====================] - loss: 0.3094 - acc: 0.9156 - val_loss: 0.2055 - val_acc: 0.9612
# 1000000/1000000 [=========================] - 32s 32us/sample - loss: 3.6479 - acc: 0.6302
# acc : [3.6479196043987274, 0.630223]

# test set 정렬했을 때
# Epoch 77/100
# 20008/20008 [=====================] - loss: 0.3162 - acc: 0.9103 - val_loss: 0.1986 - val_acc: 0.9594
# 1000000/1000000 [=========================] - 34s 34us/sample - loss: 0.2128 - acc: 0.9573
# acc : [0.21279318866223096, 0.957272]


# --------------------------------------------------------- #

# imblearn 기본
def show_balance_1(x_train, y_train, x_test, y_test):
    print(collections.Counter(y_train))
    # Counter({0: 12493, 1: 10599, 2: 1206, 3: 513, 4: 93,
    #          5: 54, 6: 36, 7: 6, 9: 5, 8: 5})

    ros = over_sampling.RandomOverSampler()
    x_train, y_train = ros.fit_resample(x_train, y_train)
    print(collections.Counter(y_train))
    # Counter({9: 12493, 8: 12493, 1: 12493, 0: 12493, 4: 12493,
    #          3: 12493, 2: 12493, 5: 12493, 6: 12493, 7: 12493})

    show_accuracy(x_train, y_train, x_test, y_test)
    # acc : [1.315881861038208, 0.308487]


# imblearn strategy 옵션
def show_balance_2(x_train, y_train, x_test, y_test):
    print(collections.Counter(y_train))
    # Counter({0: 12493, 1: 10599, 2: 1206, 3: 513, 4: 93, 5: 54, 6: 36, 7: 6, 9: 5, 8: 5})

    # strategy = {0: 12493 * 2, 1: 10599 * 2, 2: 1206 * 2, 3: 513 * 2,
    #             4: 93 * 2, 5: 54 * 2, 6: 36 * 2, 7: 6 * 2, 9: 5 * 2, 8: 5 * 2}
    strategy = {k: v*2 for k, v in collections.Counter(y_train).items()}

    ros = over_sampling.RandomOverSampler(random_state=0, sampling_strategy=strategy)
    x_train, y_train = ros.fit_resample(x_train, y_train)
    print(collections.Counter(y_train))
    # Counter({0: 24986, 1: 21198, 2: 2412, 3: 1026, 4: 186,
    #          5: 108, 6: 72, 7: 12, 9: 10, 8: 10})

    show_accuracy(x_train, y_train, x_test, y_test)
    # acc : [0.9883906559410095, 0.493381]


# show_balance_1(x_train, y_train, x_test, y_test)
# show_balance_2(x_train, y_train, x_test, y_test)
