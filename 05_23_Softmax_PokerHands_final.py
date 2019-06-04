# 처음 코드와 달라진 부분
# 1. 드롭아웃 사용 안함
#    노이즈를 인정하지 않는 데이터셋이기 때문에.
# 2. tanh 사용. relu보다 훨씬 잘 나온다
#    레이어가 깊지 않기 때문에 꼭 relu일 필요는 없다.
# 3. 정렬
#    아래 코드에서는 정렬을 하는 것이 훨씬 의미 있는 결과를 낸다.
# 4. 노드 개수
#    deep and wide에서 wide에 대해 배웠다.
#    dense에 전달되는 출력 개수가 작으면 성능은 빨리 좋아지지만 멀리 가지 못한다.
#    반면 큰 경우에는 좋아지는 것 같지 않지만 좋아진 이후에는 더 오래 간다.
#    더 많은 에포크에 대해 오버피팅 없이 성능이 향상된다.
#    2048까지 테스트
#    그러나 크기에 비례해서 학습 시간 또한 기하급수적으로 늘어난다.
# 5. 배치 크기
#    큰 영향을 주는 것 같지는 않다.
#    32, 64, 128에서 약간의 차이는 있었지만, 결정적이지는 않은 느낌.
# 6. 검증 비율
#    validation_split을 지정할 때 작게 줄수록 좋은 성능이 나온다.
#    데이터의 개수가 많을 수록 잘 나오는 것이야말로 정석이라는 것을 보여준다.
#    0.3으로 시작해서 최종적으로는 0.1 전달.
# 7. 조기 종료
#    노드 개수가 작을 때는 10~15도 괜찮지만
#    1024나 2014처럼 많아지면 10 이하로 주면 좋겠다. 최적의 성능 위치를 지난 느낌.
#    물론 최고 성능을 저장할 수 있는 방법이 있다.
# 8. 중복 데이터 제거
#    정렬을 하게 되면 중복 데이터가 발생하는데 제거하면 좋았을 것을..
#    시간이 부족해서 이 부분은 검증하지 못함.


import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing


def sort_poker(x):
    tuples = [[(i[j], i[j+1]) for j in range(0, len(i), 2)] for i in x]
    tuples = [sorted(i, key=lambda t: t[0] * 100 + t[1]) for i in tuples]
    x = [[k for j in i for k in j] for i in tuples]

    return np.int32(x)


def load_poker_hands(file_path, training):
    names = ['S1', 'R1', 'S2', 'R2', 'S3', 'R3', 'S4', 'R4', 'S5', 'R5', 'CLASS']
    hands = pd.read_csv(file_path, sep=',', header=None, names=names)

    y = hands['CLASS'].values
    x = hands.drop(['CLASS'], axis=1).values

    if training:
        x = sort_poker(x)

    # 원핫 없이 표준화만 사용했는데, 효과 없었다.
    # x = preprocessing.StandardScaler().fit_transform(x)

    x = preprocessing.OneHotEncoder(sparse=False).fit_transform(x)
    x = np.int32(x)

    return x, y


def build_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(2048, activation='tanh', input_shape=input_shape))
    # model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2048, activation='tanh'))
    # model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])
    return model


def show_accuracy(x_train, y_train, x_test, y_test):
    model = build_model([x_train.shape[1]])

    # EarlyStopping 기본은 val_loss.
    callbacks = tf.keras.callbacks.EarlyStopping(patience=15)
    model.fit(x_train, y_train, batch_size=64, epochs=1000, callbacks=[callbacks],
              validation_split=0.1)

    print('acc :', model.evaluate(x_test, y_test))


x_train, y_train = load_poker_hands('poker/poker-hand-training-true.data', True)
x_test , y_test  = load_poker_hands('poker/poker-hand-testing.data', True)

show_accuracy(x_train, y_train, x_test, y_test)

# (64, 64) tanh adam(0.0001) es(5) bs(32) epochs(1000) sort valid(0.2)
# Epoch 228/1000
# 20008/20008 - loss: 0.0101 - acc: 0.9988 - val_loss: 0.0331 - val_acc: 0.9918
# 1000000/1000000 - loss: 0.0389 - acc: 0.9898
# acc : [0.038941734451904894, 0.989792]

# (128, 128) tanh adam(0.0001) es(5) bs(32) epochs(1000) sort valid(0.2)
# Epoch 187/1000
# 20008/20008 - loss: 0.0012 - acc: 1.0000 - val_loss: 0.0196 - val_acc: 0.9964
# 1000000/1000000 - loss: 0.0249 - acc: 0.9945
# acc : [0.02492310400270112, 0.994518]

# (256, 256) tanh adam(0.0001) es(5) bs(32) epochs(1000) sort valid(0.2)
# Epoch 146/1000
# 20008/20008 - loss: 6.0506e-04 - acc: 1.0000 - val_loss: 0.0287 - val_acc: 0.9944
# 1000000/1000000 - loss: 0.0352 - acc: 0.9937
# acc : [0.03516971290110005, 0.993731]

# (256, 256) tanh adam(0.0001) es(15) bs(32) epochs(1000) sort valid(0.2)
# Epoch 168/1000
# 20008/20008 - loss: 5.6232e-05 - acc: 1.0000 - val_loss: 0.0278 - val_acc: 0.9956
# 1000000/1000000 - loss: 0.0330 - acc: 0.9946
# acc : [0.03297158807616809, 0.994581]

# (512, 512) tanh adam(0.0001) es(15) bs(32) epochs(1000) sort valid(0.2)
# Epoch 120/1000
# 20008/20008 - loss: 6.2343e-05 - acc: 1.0000 - val_loss: 0.0304 - val_acc: 0.9954
# 1000000/1000000 - loss: 0.0354 - acc: 0.9949
# acc : [0.03540878439572296, 0.994892]

# (1024, 1024) tanh adam(0.0001) es(15) bs(32) epochs(1000) sort valid(0.2)
# Epoch 88/1000
# 20008/20008 - loss: 1.6139e-04 - acc: 1.0000 - val_loss: 0.0264 - val_acc: 0.9968
# 1000000/1000000 - loss: 0.0246 - acc: 0.9964
# acc : [0.024646635785138003, 0.996361]

# (2048, 2048) tanh adam(0.0001) es(15) bs(32) epochs(1000) sort valid(0.2)
# Epoch 9/1000
# 20008/20008 - loss: 0.9652 - acc: 0.4977 - val_loss: 0.9670 - val_acc: 0.4878
# 실패. 오버피팅은 아니지만, 코스트가 떨어지지 않고 정확도도 향상되지 않는다

# (256, 256) tanh adam(0.0001) es(15) bs(64) epochs(1000) sort valid(0.2)
# Epoch 242/1000
# 20008/20008 - loss: 7.7154e-05 - acc: 1.0000 - val_loss: 0.0328 - val_acc: 0.9942
# 1000000/1000000 - loss: 0.0360 - acc: 0.9940
# acc : [0.0360011936801119, 0.994014]

# (256, 256) tanh adam(0.0001) es(15) bs(128) epochs(1000) sort valid(0.2)
# Epoch 328/1000
# 20008/20008 - loss: 1.0875e-04 - acc: 1.0000 - val_loss: 0.0197 - val_acc: 0.9956
# 1000000/1000000 - loss: 0.0300 - acc: 0.9946
# acc : [0.029964904714002885, 0.994626]

# (256, 256) tanh adam(0.0001) es(15) bs(128) epochs(1000) sort valid(0.1)
# Epoch 328/1000
# 22509/22509 - loss: 3.5107e-05 - acc: 1.0000 - val_loss: 0.0166 - val_acc: 0.9960
# 1000000/1000000 - loss: 0.0237 - acc: 0.9958
# acc : [0.023729507474604266, 0.995809]

# (512, 512) tanh adam(0.0001) es(15) bs(128) epochs(1000) sort valid(0.1)
# Epoch 198/1000
# 22509/22509 - loss: 4.0726e-04 - acc: 1.0000 - val_loss: 0.0209 - val_acc: 0.9968
# 1000000/1000000 - loss: 0.0284 - acc: 0.9956
# acc : [0.028373807210865198, 0.995572]

# (1024, 1024) tanh adam(0.0001) es(15) bs(128) epochs(1000) sort valid(0.1)
# Epoch 142/1000
# 22509/22509 - loss: 4.8238e-04 - acc: 1.0000 - val_loss: 0.0196 - val_acc: 0.9980
# 1000000/1000000 - loss: 0.0209 - acc: 0.9969
# acc : [0.020944544215374162, 0.996878]

# (1024, 1024) tanh adam(0.0001) es(15) bs(64) epochs(1000) sort valid(0.1)
# Epoch 109/1000
# 22509/22509 - loss: 1.3132e-04 - acc: 1.0000 - val_loss: 0.0195 - val_acc: 0.9980
# 1000000/1000000 - loss: 0.0218 - acc: 0.9968
# acc : [0.021840463847288164, 0.996813]

# (2048, 2048) tanh adam(0.0001) es(15) bs(64) epochs(1000) sort valid(0.1)
# Epoch 103/1000
# 22509/22509 - loss: 1.1605e-04 - acc: 1.0000 - val_loss: 0.0178 - val_acc: 0.9980
# 1000000/1000000 - 188s 188us/sample - loss: 0.0203 - acc: 0.9967
# acc : [0.020288726077554515, 0.996727]
