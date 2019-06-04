import tensorflow as tf
import numpy as np


# cnn 모델 : 마지막 레이어를 fc로 구성
def build_model_fc():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(128, [3, 3], padding='same', input_shape=[28, 28, 1]))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv2D(128, [3, 3], padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPool2D([2, 2]))

    model.add(tf.keras.layers.Conv2D(256, [3, 3], padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv2D(256, [3, 3], padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPool2D([2, 2]))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model


# cnn 모델 : 마지막 레이어를 fc 대신 1x1 컨볼루션으로 구성
# fc에 비해 성능이 떨어진다. 이미지 크기가 커서 레이어를 여러 개 쌓을 수 있을 때 다시 검증할 것.
def build_model_conv():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(128, [3, 3], padding='same', input_shape=[28, 28, 1]))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv2D(128, [3, 3], padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPool2D([2, 2]))

    model.add(tf.keras.layers.Conv2D(256, [3, 3], padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv2D(256, [3, 3], padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPool2D([2, 2]))

    # fc ==> 1x1 컨볼루션. Flatten 레이어를 뒤쪽에 배치
    model.add(tf.keras.layers.Conv2D(512, [7, 7], padding='same'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv2D(512, [1, 1], padding='same'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model


def cnn_mnist_fashion(model_func, filename):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test  = x_test .reshape(-1, 28, 28, 1)

    x_train = x_train / 255     # 스케일링. 0~1 사이의 데이터로 변환
    x_test  = x_test  / 255

    # ----------------------------------------------------- #

    model = model_func()

    # sparse 버전 사용
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001, decay=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    # 가장 좋았을 때의 가중치 저장은 오히려 결과가 좋지 않았다.
    es = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=False)

    model.fit(x_train, y_train, batch_size=64, epochs=100, callbacks=[es], validation_split=0.1, verbose=2)
    print(model.evaluate(x_test, y_test))

    model.save(filename)


# 이미지 증강. 조금이긴 하지만 성능 향상.
def cnn_mnist_fashion_augmentation(model_func, filename):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test  = x_test .reshape(-1, 28, 28, 1)

    x_train = x_train / 255
    x_test  = x_test  / 255

    train_size = int(len(x_train) * 0.9)
    x_train, x_valid = x_train[:train_size], x_train[train_size:]
    y_train, y_valid = y_train[:train_size], y_train[train_size:]

    # ----------------------------------------------------- #

    model = model_func()

    # sparse 버전 사용
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001, decay=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    es = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=False)

    train_img_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)
    valid_img_generator = tf.keras.preprocessing.image.ImageDataGenerator()

    batch_size = 64

    train_generator = train_img_generator.flow(x_train, y_train, batch_size=batch_size)
    valid_generator = valid_img_generator.flow(x_valid, y_valid, batch_size=batch_size)

    model.fit_generator(train_generator,
                        steps_per_epoch=1000 // batch_size,
                        epochs=100,
                        validation_data=valid_generator,
                        validation_steps=50,
                        callbacks=[es],
                        verbose=2)

    print(model.evaluate(x_test, y_test))

    model.save(filename)


# 저장한 모델 파일을 사용해서 앙상블 구현
# 시간이 오래 걸리기 때문에 모델 파일 생성과 예측을 별도 함수로 구현
def predict_model_by_ensemble(model_path_text, n_models):
    (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_test  = x_test .reshape(-1, 28, 28, 1)
    x_test  = x_test  / 255

    binds = np.zeros([len(x_test), 10], dtype=np.float32)
    for i in range(n_models):
        model = tf.keras.models.load_model(model_path_text.format(i))

        preds = model.predict(x_test)
        binds += preds

        preds_bool = np.argmax(preds, axis=1)
        print('{} : {}'.format(i, np.mean(preds_bool == y_test)))

    print('-' * 50)

    binds_bool = np.argmax(binds, axis=1)
    print('{} : {}'.format(i, np.mean(binds_bool == y_test)))


# 1회 호출로 대략적인 정확도 검증
# cnn_mnist_fashion(build_model_fc, 'mnist_fashion_fc.h5')
# cnn_mnist_fashion(build_model_conv, 'mnist_fashion_conv.h5')
# cnn_mnist_fashion_augmentation(build_model_fc, 'mnist_fashion_aug_fc.h5')

# 앙상블에 사용하기 위한 모델 파일 생성
# for i in range(10):
#     cnn_mnist_fashion(build_model_fc, 'Model/mnist_fashion_fc_{}.h5'.format(i))
#     cnn_mnist_fashion(build_model_conv, 'Model/mnist_fashion_conv_{}.h5'.format(i))
#     cnn_mnist_fashion_augmentation(build_model_fc, 'Model/mnist_fashion_aug_fc_{}.h5'.format(i))

# 앙상블
# predict_model_by_ensemble('Model/mnist_fashion_fc_{}.h5', 10)
# predict_model_by_ensemble('Model/mnist_fashion_conv_{}.h5', 10)
# predict_model_by_ensemble('Model/mnist_fashion_aug_fc_{}.h5', 10)
