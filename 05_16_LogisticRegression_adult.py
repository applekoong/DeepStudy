import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt

pd.options.display.max_columns = 100
pd.options.display.width = 1000
np.set_printoptions(linewidth=1000)


def get_xy(adult):
    def convert_column(df, col_name):
        # df['income'] = df['income'].map({"<=50K": 0, ">50K": 1})

        uniques = df[col_name].unique()
        items = {v: i for i, v in enumerate(uniques)}

        df[col_name] = df[col_name].map(items)

    # 문자열 데이터를 숫자로 변환
    objects = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']
    for col_name in objects:
        convert_column(adult, col_name)

    # 모든 데이터가 정수 형태로 바뀐 것 확인
    # adult.info()
    # print(adult.head())

    # 데이터셋의 대략적인 분포 확인
    # print(adult.describe())

    y = adult['income'].values
    adult = adult.drop(['income'], axis=1)

    return adult.values, y


def get_xy_onehot(adult):
    income = adult['income'].values
    y = preprocessing.LabelBinarizer().fit_transform(income)
    adult = adult.drop(['income'], axis=1)

    x = []
    for col_name in adult.columns:
        feature = adult[col_name].values

        if adult[col_name].dtype == np.object:
            feature = preprocessing.LabelBinarizer().fit_transform(feature)
        else:
            feature = feature[:, np.newaxis]

        x.append(feature)

    # 리스트에 저장한 피처를 2차원 배열로 변환
    return np.concatenate(x, axis=1), y


# 연속형과 범주형 데이터를 각각 반환 (앙상블 적용하면 평균 86.5%)
def get_xy_onehot_categorical(adult):
    income = adult['income'].values
    y = preprocessing.LabelBinarizer().fit_transform(income)
    adult = adult.drop(['income'], axis=1)

    x1, x2 = [], []
    for col_name in adult.columns:
        feature = adult[col_name].values

        if adult[col_name].dtype == np.object:
            feature = preprocessing.LabelBinarizer().fit_transform(feature)
            x2.append(feature)
        else:
            feature = feature[:, np.newaxis]
            x1.append(feature)

    return np.concatenate(x1, axis=1), np.concatenate(x2, axis=1), y


# 목적
# 다양한 하이퍼파라미터 검증. 최적의 조합 검색.
def show_single_model(x_train, x_test, y_train, y_test):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu, input_shape=(x.shape[1],)))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0015),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    history = model.fit(x_train, y_train, batch_size=50, epochs=50,
                        validation_data=[x_test, y_test], verbose=0)
    return model.evaluate(x_test, y_test, verbose=0)

    # print(history.history.keys())   # ['loss', 'acc', 'val_loss', 'val_acc']

    # 오버피팅 확인
    # plt.figure()
    # plt.title('loss')
    # plt.plot(history.history['loss'], 'r')
    # plt.plot(history.history['val_loss'], 'g')
    #
    # plt.figure()
    # plt.title('acc')
    # plt.plot(history.history['acc'], 'r')
    # plt.plot(history.history['val_acc'], 'g')
    # plt.show()


# 앙상블에서만 사용
def build_and_train_model(input_shape, lr):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu, input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Dense(units=3, activation=tf.nn.relu))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])
    return model


# 앙상블 함수
def show_ensemble(x_train, x_test, y_train, y_test, batch_size, epochs, lr, n_iters):
    result = np.zeros(y_test.shape)
    for i in range(n_iters):
        model = build_and_train_model((x_train.shape[1],), lr=lr)

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_split=0.1)
        preds = model.predict(x_test)
        result += preds

        # 개별 모델 성능 표시
        print(i, np.mean((preds > 0.5) == y_test))
        print(i, np.mean((preds > 0.45) == y_test))
    print('-' * 50)

    # 5개 앙상블이라면 2.5가 기준점
    print('acc :', np.mean((result > n_iters/2) == y_test))


# 14개의 피처 중에서 10개를 선발해서 학습한 결과를 앙상블 (완전 실패. 평균 75%)
def show_ensemble_meta(x_train, x_test, y_train, y_test, batch_size, epochs, lr, n_iters):
    result = np.zeros(y_test.shape)
    for i in range(n_iters):
        attend = np.random.choice(x_train.shape[1], size=10, replace=False)
        x_train_small = x_train[:, attend]
        x_test_small = x_test[:, attend]

        model = build_and_train_model((x_train_small.shape[1],), lr=lr)

        model.fit(x_train_small, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_split=0.1)
        preds = model.predict(x_test_small)
        result += preds

        # 개별 모델 성능 표시
        print(i, np.mean((preds > 0.5) == y_test))
    print('-' * 50)

    # 5개 앙상블이라면 2.5가 기준점
    print('acc :', np.mean((result > n_iters/2) == y_test))


names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
         'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
         'hours-per-week', 'native-country', 'income']

# 컬럼 사이에 공백 있기 때문에 구분자로 ', ' 사용
adult = pd.read_csv('Data/adult.data', header=None, names=names, sep=', ', engine='python')

adult = adult.drop(['education', 'education-num', 'marital-status'], axis=1)

# 대략적인 구조 파악
# adult.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 32561 entries, 0 to 32560
# Data columns (total 15 columns):
# age               32561 non-null int64
# workclass         32561 non-null object
# fnlwgt            32561 non-null int64
# education         32561 non-null object
# education-num     32561 non-null int64
# marital-status    32561 non-null object
# occupation        32561 non-null object
# relationship      32561 non-null object
# race              32561 non-null object
# sex               32561 non-null object
# capital-gain      32561 non-null int64
# capital-loss      32561 non-null int64
# hours-per-week    32561 non-null int64
# native-country    32561 non-null object
# income            32561 non-null object
# dtypes: int64(6), object(9)
# memory usage: 3.7+ MB

# x, y = get_xy(adult)
# x, y = get_xy_onehot(adult)
# 
# x = preprocessing.minmax_scale(x)
# x = preprocessing.StandardScaler().fit_transform(x)     # minmax와 비슷한 결과

# 범주형 데이터는 정규화에서 제외
x1, x2, y = get_xy_onehot_categorical(adult)

x1 = preprocessing.minmax_scale(x1)
x = np.concatenate([x1, x2], axis=1)

# np.random.seed(100)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)
# print(x_train.shape, x_test.shape)      # (22792, 14) (9769, 14)
# print(y_train.shape, y_test.shape)      # (22792,) (9769,)

# print(x_train[0])
# [0.38356164 0.25 0.18525624 0. 0.8 0.16666667 0.07142857 0.2 0. 0. 0. 0. 0.5 0.]

# ----------------------------------- #

# for i in range(3):
#     print(i, show_single_model(x_train, x_test, y_train, y_test))

show_ensemble(x_train, x_test, y_train, y_test, batch_size=100, epochs=50, lr=0.002, n_iters=3)
# show_ensemble_meta(x_train, x_test, y_train, y_test, batch_size=100, epochs=50, lr=0.002, n_iters=3)

# 레이어 1개, 에포크 20, 배치 100
# acc : [0.38539799375804473, 0.82894874]

# 레이어 2개, 에포크 50, 배치 100
# acc : [0.36362122439937883, 0.84553176]

# 레이어 1개, 에포크 30, 배치 50, 배치놈 : 레이어 늘리면 안 좋다
# acc : [0.32841471733584693, 0.84962636]
# acc : [0.33428295724698714, 0.85075235]   배치 100개

# acc : [0.3241405077045998, 0.8535162]
# loss가 0.31까지 떨어지면 85.4% 정확도 달성

# 0 0.8509571092230526
# 1 0.8512642030914116
# 2 0.8521854846964889
# --------------------------------------------------
# acc : 0.8545398710205753

# 0 0.8568942573446617
# 1 0.8555635172484389
# 2 0.8555635172484389
# --------------------------------------------------
# acc : 0.8567918927218753

# 0 0.8555635172484389
# 1 0.8573037158358071
# 2 0.8569966219674481
# --------------------------------------------------
# acc : 0.8573037158358071

# 0 0.8554611526256526
# 1 0.8582249974408844
# 2 0.8568942573446617
# --------------------------------------------------
# acc : 0.8582249974408844
