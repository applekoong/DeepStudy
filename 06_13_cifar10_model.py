import tensorflow as tf
import numpy as np
import collections
import time


def build_model_fc():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(128, [3, 3], padding='same', input_shape=[32, 32, 3]))
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


def cifar10_basic(x_train, y_train, x_test, y_test, filename):
    model = build_model_fc()

    # sparse 버전 사용
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001, decay=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    # 가장 좋았을 때의 가중치 저장은 오히려 결과가 좋지 않았다.
    es = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=False)

    model.fit(x_train, y_train, batch_size=64, epochs=100, callbacks=[es], validation_split=0.1, verbose=2)
    print(model.evaluate(x_test, y_test))

    model.save(filename)


def cifar10_basic_augmentation_many(x_train, y_train, x_test, y_test, filename):
    train_size = int(len(x_train) * 0.9)
    x_train, x_valid = x_train[:train_size], x_train[train_size:]
    y_train, y_valid = y_train[:train_size], y_train[train_size:]

    # ----------------------------------------------------- #

    model = build_model_fc()

    # sparse 버전 사용
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001, decay=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    es = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=False)

    train_img_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                                                                          rotation_range=20,
                                                                          width_shift_range=0.1,
                                                                          height_shift_range=0.1,
                                                                          shear_range=0.1,
                                                                          zoom_range=0.1)
    valid_img_generator = tf.keras.preprocessing.image.ImageDataGenerator()

    batch_size = 64

    train_generator = train_img_generator.flow(x_train, y_train, batch_size=batch_size)
    valid_generator = valid_img_generator.flow(x_valid, y_valid, batch_size=batch_size)

    model.fit_generator(train_generator,
                        steps_per_epoch=len(x_train) // batch_size,
                        epochs=100,
                        validation_data=valid_generator,
                        callbacks=[es],
                        verbose=2)

    print(model.evaluate(x_test, y_test))

    model.save(filename)


# mnist_fashion에서 사용한 코드 재사용
def build_model_vgg16():
    conv_base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=[32, 32, 3])

    model = tf.keras.models.Sequential()
    model.add(conv_base)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model


# 증강 옵션으로 mnist_fashion과 동일하게 수평 뒤집기만 적용. 1/255 스케일링 적용
def cifar10_vgg16_augmentation(x_train, y_train, x_test, y_test, filename):
    train_size = int(len(x_train) * 0.9)
    x_train, x_valid = x_train[:train_size], x_train[train_size:]
    y_train, y_valid = y_train[:train_size], y_train[train_size:]

    # ----------------------------------------------------- #

    model = build_model_fc()

    # sparse 버전 사용
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001, decay=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    es = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=False)

    train_img_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, rescale=1/255)
    valid_img_generator = tf.keras.preprocessing.image.ImageDataGenerator()

    batch_size = 64

    train_generator = train_img_generator.flow(x_train, y_train, batch_size=batch_size)
    valid_generator = valid_img_generator.flow(x_valid, y_valid, batch_size=batch_size)

    model.fit_generator(train_generator,
                        steps_per_epoch=len(x_train) // batch_size,
                        epochs=100,
                        validation_data=valid_generator,
                        callbacks=[es],
                        verbose=2)

    print(model.evaluate(x_test, y_test))

    model.save(filename)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train / np.float32(255)  # 스케일링. 0~1 사이의 데이터로 변환
x_test = x_test / np.float32(255)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

print(collections.Counter(y_train.reshape(-1)))

print(x_train[0][0][:5])
print(x_train.dtype)

model = build_model_fc()
model.summary()

start = time.time()
# cifar10_basic(x_train, y_train, x_test, y_test, 'cifar10_basic.h5')
cifar10_basic_augmentation_many(x_train, y_train, x_test, y_test, 'cifar10_basic.h5')
print('elasped :', time.time() - start, 'seconds')
