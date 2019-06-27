# uci arabic audio 데이터셋
# https://archive.ics.uci.edu/ml/datasets/Spoken+Arabic+Digit

# 1. 오디오로부터 피처를 추출한 mfcc 텍스트 파일
# 2. 학습/시험 : 75/25
#    train : 6,600
#    test  : 2,200
# 3. Dataset from 8800(10 digits x 10 repetitions x 88 speakers) time series of 13 Frequency Cepstral
#    Coefficients (MFCCs) had taken from 44 males and 44 females Arabic native speakers
#    between the ages 18 and 40 to represent ten spoken Arabic digit.
# 4. Class Distribution:
#    Each line in Train_Arabic_Digit.txt or Test_Arabic_Digit.txt represents 13 MFCCs coefficients in
#    the increasing order separated by spaces. This corresponds to one analysis frame.
#    발성은 mfcc 계수가 증가하는 형태로 저장되고, 각각의 분석 프레임에 대응된다.
#    Lines are organized into blocks, which are a set of 4-93 lines separated by blank lines and
#    corresponds to a single speech utterance of an spoken Arabic digit with 4-93 frames.
#    빈 줄로 구분되는 블록은 최소 4줄에서 최대 93줄까지로 구성되고, 각각은 발성 1개를 나타낸다.
#    Each spoken digit is a set of consecutive blocks.
#    발음된 숫자는 연속된 블록에 들어있다.
#    In Train_Arabic_Digit.txt there are 660 blocks for each spoken digit.
#    학습 데이터셋에는 각각의 숫자를 가리키는 660개의 블록들이 순서대로 온다.
#    The first 330 blocks represent male speakers and the second 330 blocks represent the female speakers.
#    각각의 숫자에 대해 330개의 남성 블록이 먼저 오고, 330개의 여성 블록이 나중에 오는 방식으로 구성된다.
#    Blocks 1-660 represent the spoken digit "0" (10 utterances of /0/ from 66 speakers),
#    blocks 661-1320 represent the spoken digit "1" (10 utterances of /1/
#    from the same 66 speakers 33 males and 33 females), and so on up to digit 9.
#    0~9까지의 숫자가 순서대로 660개의 블록으로 구성된다.
#    0이 660개, 1이 660개, 2가 660개의 순서.
#    화자가 0~9를 한 번에 발성한 것이 아니라 66명의 화자가 발음한 0을 모아놓았다.
#    Speakers in the test dataset are different from those in the train dataset.
#    시험 데이터셋은 22명의 화자가 참여한 것외에는 학습 데이터셋과 동일하다.
#    다만, 학습 데이터셋에 참여하지 않았던 화자들로 구성된다.


import keras
import numpy as np


# 프레임 개수 세기. mfcc 벡터 사이에 빈 줄이 한 줄씩 들어있다.
def frame_count():
    # f = open('audio/Train_Arabic_Digit.txt', 'r', encoding='utf-8')   # 6600
    f = open('audio/Test_Arabic_Digit.txt', 'r', encoding='utf-8')      # 2200

    blank = 0
    for line in f:
        line = line.strip()

        if not line:
            blank += 1

    print(blank)


# 프레임에 포함된 줄 수 세기.
# 4줄에서 93줄까지 너무 차이가 나서 좋은 결과를 내는 것이 쉽지 않아 보임.
def frame_min_max():
    f = open('audio/Train_Arabic_Digit.txt', 'r', encoding='utf-8')
    # f = open('audio/Test_Arabic_Digit.txt', 'r', encoding='utf-8')

    blocks, frame = [], []
    for line in f:
        line = line.strip()

        if not line:
            if frame:
                blocks.append(frame)
            frame = []
        else:
            frame.append(line)

    line_counts = [len(i) for i in blocks]
    print('min :', min(line_counts))        # train(4), test(7)
    print('max :', max(line_counts))        # train(93), test(83)


#
def make_features_and_labels_by_digit(filename, shuffle):
    f = open(filename, 'r', encoding='utf-8')

    features, frame = [], []
    for line in f:
        line = line.strip()

        # 빈 줄이라면, 새로운 프레임의 시작
        if not line:
            # 프레임에 저장된 내용이 있으면 피처에 추가
            if frame:
                # 줄 수가 모두 다르기 때문에 평균 처리
                frame_mean = np.mean(np.float32(frame), axis=0)
                features.append(frame_mean)
            frame = []
        else:
            frame.append(line.split())

    # 마지막 프레임 추가
    frame_mean = np.mean(np.float32(frame), axis=0)
    features.append(frame_mean)

    # ------------------------------------------ #

    n_classes = 10
    assert len(features) % n_classes == 0
    blocks_per_gender = len(features) // n_classes

    labels = []
    for i in range(n_classes):
        labels += [i] * blocks_per_gender

    assert len(features) == len(labels)     # 6600 or 2200

    features = np.float32(features)
    labels = np.float32(labels)

    # 피처는 gender가 됐건 digit이 됐건 똑같은 데이터셋이고
    # 레이블은 크기는 같지만 gender는 0과 1, digit은 0~9 사이의 정수가 된다.
    # train (6600, 13) (6600,)
    # test  (2200, 13) (2200,)
    # print(features.shape, labels.shape)

    if shuffle:
        indecies = np.arange(len(features))
        np.random.shuffle(indecies)

        features = features[indecies]
        labels = labels[indecies]

    # print(np.unique(labels))      # [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    return features, labels


def make_features_and_labels_by_gender(filename, shuffle):
    f = open(filename, 'r', encoding='utf-8')

    features, frame = [], []
    for line in f:
        line = line.strip()

        if not line:
            if frame:
                frame_mean = np.mean(np.float32(frame), axis=0)
                features.append(frame_mean)
            frame = []
        else:
            frame.append(line.split())

    # 마지막 프레임 추가
    frame_mean = np.mean(np.float32(frame), axis=0)
    features.append(frame_mean)

    # ------------------------------------------ #

    n_classes = 2
    assert len(features) % n_classes == 0
    blocks_per_gender = len(features) // n_classes

    labels = []
    for i in range(n_classes):
        labels += [i] * blocks_per_gender

    assert len(features) == len(labels)

    features = np.float32(features)
    labels = np.float32(labels)

    # 피처는 gender가 됐건 digit이 됐건 똑같은 데이터셋이고
    # 레이블은 크기는 같지만 gender는 0과 1, digit은 0~9 사이의 정수가 된다.
    # train (6600, 13) (6600,)
    # test  (2200, 13) (2200,)
    # print(features.shape, labels.shape)

    if shuffle:
        indecies = np.arange(len(features))
        np.random.shuffle(indecies)

        features = features[indecies]
        labels = labels[indecies]

    # print(np.unique(labels))      # [0. 1.]
    return features, labels


def get_arabic_digits_dataset_by_gender():
    x_train, y_train = make_features_and_labels_by_gender('audio/Train_Arabic_Digit.txt', shuffle=True)
    x_test, y_test = make_features_and_labels_by_gender('audio/Test_Arabic_Digit.txt', shuffle=False)

    return x_train, y_train, x_test, y_test


def get_arabic_digits_dataset_by_digit():
    x_train, y_train = make_features_and_labels_by_digit('audio/Train_Arabic_Digit.txt', shuffle=True)
    x_test, y_test = make_features_and_labels_by_digit('audio/Test_Arabic_Digit.txt', shuffle=False)

    return x_train, y_train, x_test, y_test


def show_digits_by_gender():
    x_train, y_train, x_test, y_test = get_arabic_digits_dataset_by_gender()

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(128, activation='relu', input_shape=[13]))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['acc'])

    early_stop = keras.callbacks.EarlyStopping(patience=5)

    history = model.fit(x_train, y_train, epochs=1000, batch_size=128, verbose=2, callbacks=[early_stop],
                        validation_split=0.2)

    print(model.evaluate(x_test, y_test))


def show_digits_by_digit():
    x_train, y_train, x_test, y_test = get_arabic_digits_dataset_by_digit()

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(10, activation='softmax', input_shape=[13]))

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    early_stop = keras.callbacks.EarlyStopping(patience=5)

    history = model.fit(x_train, y_train, epochs=1000, batch_size=128, verbose=2, callbacks=[early_stop],
                        validation_split=0.2)

    print(model.evaluate(x_test, y_test))


# frame_count()
# frame_min_max()

# show_digits_by_gender()     # [0.46859066657044673, 0.8154545454545454]
# show_digits_by_digit()      # [0.8150064208290794, 0.6968181818181818]
