# Resnet 구현
# 1. 레이어 34와 50 구현
# 2. 레이어 34는 1x1 컨볼루션을 사용하지 않는다.
#    숏컷 커넥션 구현하지 않았고, 구성도에 맞게 shape이 바뀌는 것만 확인
# 3. 레이어 50을 구현한 목적은 케라스에서 제공하는 유일한 모델이어서.
#    shape을 일치시키는 단순한 버전부터 파라미터 개수까지 정확하게 일치시키는 최종 버전까지 구현
# 4. cifar10에 적용하면 결과가 전혀 안 나온다.
#    하이퍼파라미터를 조절해도 30% 넘기가 쉽지 않다.
#    크기가 너무 작아서 1x1 컨볼루션과 3x3 필터 중첩이 동작하지 않는 것처럼 보인다.
# 5. vgg 기본 코드에 shortcut 연결

import keras


def resnet_34_simple():
    input_tensor = keras.layers.Input([224, 224, 3])
    output = input_tensor

    # conv_1 --------------------- #
    output = keras.layers.Conv2D(64, [7, 7], padding='same', strides=[2, 2])(output)

    # conv_2 --------------------- #
    output = keras.layers.MaxPool2D([3, 3], padding='same', strides=[2, 2])(output)
    output = keras.layers.Conv2D(64, [3, 3], padding='same')(output)
    output = keras.layers.Conv2D(64, [3, 3], padding='same')(output)
    output = keras.layers.Conv2D(64, [3, 3], padding='same')(output)

    # conv_3 --------------------- #
    output = keras.layers.Conv2D(128, [3, 3], padding='same', strides=[2, 2])(output)
    output = keras.layers.Conv2D(128, [3, 3], padding='same')(output)
    output = keras.layers.Conv2D(128, [3, 3], padding='same')(output)
    output = keras.layers.Conv2D(128, [3, 3], padding='same')(output)

    # conv_4 --------------------- #
    output = keras.layers.Conv2D(256, [3, 3], padding='same', strides=[2, 2])(output)
    output = keras.layers.Conv2D(256, [3, 3], padding='same')(output)
    output = keras.layers.Conv2D(256, [3, 3], padding='same')(output)
    output = keras.layers.Conv2D(256, [3, 3], padding='same')(output)
    output = keras.layers.Conv2D(256, [3, 3], padding='same')(output)
    output = keras.layers.Conv2D(256, [3, 3], padding='same')(output)

    # conv_5 --------------------- #
    output = keras.layers.Conv2D(512, [3, 3], padding='same', strides=[2, 2])(output)
    output = keras.layers.Conv2D(512, [3, 3], padding='same')(output)
    output = keras.layers.Conv2D(512, [3, 3], padding='same')(output)

    # fc --------------------- #
    output = keras.layers.GlobalAveragePooling2D()(output)
    output = keras.layers.Dense(1000)(output)

    model = keras.models.Model(inputs=input_tensor, outputs=output)
    model.summary()


def resnet_50_simple():
    input_tensor = keras.layers.Input([224, 224, 3])
    output = input_tensor

    # conv_1 --------------------- #
    output = keras.layers.Conv2D(64, [7, 7], padding='same', strides=[2, 2])(output)

    # conv_2 --------------------- #
    output = keras.layers.MaxPool2D([3, 3], padding='same', strides=[2, 2])(output)

    for _ in range(3):
        output = keras.layers.Conv2D(64, [1, 1], padding='same')(output)
        output = keras.layers.Conv2D(64, [3, 3], padding='same')(output)
        output = keras.layers.Conv2D(256, [1, 1], padding='same')(output)

    # conv_3 --------------------- #
    for i in range(4):
        strides = [2, 2] if i == 0 else [1, 1]

        output = keras.layers.Conv2D(128, [1, 1], padding='same', strides=strides)(output)
        output = keras.layers.Conv2D(128, [3, 3], padding='same')(output)
        output = keras.layers.Conv2D(512, [1, 1], padding='same')(output)

    # conv_4 --------------------- #
    for i in range(6):
        strides = [2, 2] if i == 0 else [1, 1]

        output = keras.layers.Conv2D(256, [1, 1], padding='same', strides=strides)(output)
        output = keras.layers.Conv2D(256, [3, 3], padding='same')(output)
        output = keras.layers.Conv2D(1024, [1, 1], padding='same')(output)

    # conv_5 --------------------- #
    for i in range(3):
        strides = [2, 2] if i == 0 else [1, 1]

        output = keras.layers.Conv2D(512, [1, 1], padding='same', strides=strides)(output)
        output = keras.layers.Conv2D(512, [3, 3], padding='same')(output)
        output = keras.layers.Conv2D(2048, [1, 1], padding='same')(output)

    # fc --------------------- #
    output = keras.layers.GlobalAveragePooling2D()(output)
    output = keras.layers.Dense(1000)(output)

    model = keras.models.Model(inputs=input_tensor, outputs=output)
    model.summary()


def resnet_50_shortcut():
    input_tensor = keras.layers.Input([224, 224, 3])
    output = input_tensor

    # conv_1 --------------------- #
    output = keras.layers.Conv2D(64, [7, 7], padding='same', strides=[2, 2])(output)
    output = keras.layers.MaxPool2D([3, 3], padding='same', strides=[2, 2])(output)

    # conv_2 --------------------- #
    strides = 1
    shortcut = keras.layers.Conv2D(256, [1, 1], padding='same', strides=strides)(output)

    for i in range(3):
        output = keras.layers.Conv2D(64, [1, 1], padding='same', strides=strides)(output)
        output = keras.layers.Conv2D(64, [3, 3], padding='same')(output)
        output = keras.layers.Conv2D(256, [1, 1], padding='same')(output)

        output = keras.layers.Add()([output, shortcut])
        shortcut = output
        strides = 1

    # conv_3 --------------------- #
    strides = 2
    shortcut = keras.layers.Conv2D(512, [1, 1], padding='same', strides=strides)(output)

    for i in range(4):
        output = keras.layers.Conv2D(128, [1, 1], padding='same', strides=strides)(output)
        output = keras.layers.Conv2D(128, [3, 3], padding='same')(output)
        output = keras.layers.Conv2D(512, [1, 1], padding='same')(output)

        output = keras.layers.Add()([output, shortcut])
        shortcut = output
        strides = 1

    # conv_4 --------------------- #
    strides = 2
    shortcut = keras.layers.Conv2D(1024, [1, 1], padding='same', strides=strides)(output)

    for i in range(6):
        output = keras.layers.Conv2D(256, [1, 1], padding='same', strides=strides)(output)
        output = keras.layers.Conv2D(256, [3, 3], padding='same')(output)
        output = keras.layers.Conv2D(1024, [1, 1], padding='same')(output)

        output = keras.layers.Add()([output, shortcut])
        shortcut = output
        strides = 1

    # conv_5 --------------------- #
    strides = 2
    shortcut = keras.layers.Conv2D(2048, [1, 1], padding='same', strides=strides)(output)

    for i in range(3):
        output = keras.layers.Conv2D(512, [1, 1], padding='same', strides=strides)(output)
        output = keras.layers.Conv2D(512, [3, 3], padding='same')(output)
        output = keras.layers.Conv2D(2048, [1, 1], padding='same')(output)

        output = keras.layers.Add()([output, shortcut])
        shortcut = output
        strides = 1

    # fc --------------------- #
    output = keras.layers.GlobalAveragePooling2D()(output)
    output = keras.layers.Dense(1000)(output)

    model = keras.models.Model(inputs=input_tensor, outputs=output)
    model.summary()


def resnet_50_bottle_neck_1():
    def residual_blocks(output, n_filters, strides, n_blocks):
        shortcut = keras.layers.Conv2D(n_filters * 4, [1, 1], padding='same', strides=strides)(output)

        for _ in range(n_blocks):
            output = bottle_neck(output, n_filters, strides)

            output = keras.layers.Add()([output, shortcut])
            shortcut = output
            strides = 1

        return output

    def bottle_neck(output, n_filters, strides):
        output = keras.layers.Conv2D(n_filters, [1, 1], padding='same', strides=strides)(output)
        output = keras.layers.Conv2D(n_filters, [3, 3], padding='same')(output)
        output = keras.layers.Conv2D(n_filters * 4, [1, 1], padding='same')(output)
        return output

    input_tensor = keras.layers.Input([224, 224, 3])
    output = input_tensor

    # conv_1 --------------------- #
    output = keras.layers.Conv2D(64, [7, 7], padding='same', strides=[2, 2])(output)
    output = keras.layers.MaxPool2D([3, 3], padding='same', strides=[2, 2])(output)

    # conv 2, 3, 4, 5 ------------ #
    output = residual_blocks(output, n_filters=64, strides=1, n_blocks=3)
    output = residual_blocks(output, n_filters=128, strides=2, n_blocks=4)
    output = residual_blocks(output, n_filters=256, strides=2, n_blocks=6)
    output = residual_blocks(output, n_filters=512, strides=2, n_blocks=3)

    # fc --------------------- #
    output = keras.layers.GlobalAveragePooling2D()(output)
    output = keras.layers.Dense(1000)(output)

    model = keras.models.Model(inputs=input_tensor, outputs=output)
    model.summary()
    model.summary()


def resnet_50_bottle_neck_2():
    def residual_blocks(output, n_filters, strides, n_blocks):
        shortcut = keras.layers.Conv2D(n_filters * 4, [1, 1], padding='same', strides=strides)(output)
        shortcut = keras.layers.BatchNormalization()(shortcut)

        for _ in range(n_blocks):
            output = bottle_neck(output, n_filters, strides)

            output = keras.layers.Add()([output, shortcut])
            output = keras.layers.ReLU()(output)

            shortcut = output
            strides = 1

        return output

    def bottle_neck(output, n_filters, strides):
        output = keras.layers.Conv2D(n_filters, [1, 1], padding='same', strides=strides)(output)
        output = keras.layers.BatchNormalization()(output)
        output = keras.layers.ReLU()(output)

        output = keras.layers.Conv2D(n_filters, [3, 3], padding='same')(output)
        output = keras.layers.BatchNormalization()(output)
        output = keras.layers.ReLU()(output)

        output = keras.layers.Conv2D(n_filters * 4, [1, 1], padding='same')(output)
        output = keras.layers.BatchNormalization()(output)
        return output

    input_tensor = keras.layers.Input([224, 224, 3])
    output = input_tensor

    # conv_1 --------------------- #
    output = keras.layers.Conv2D(64, [7, 7], padding='same', strides=[2, 2])(output)
    output = keras.layers.BatchNormalization()(output)
    output = keras.layers.ReLU()(output)

    output = keras.layers.MaxPool2D([3, 3], padding='same', strides=[2, 2])(output)

    # conv 2, 3, 4, 5 ------------ #
    output = residual_blocks(output, n_filters=64, strides=1, n_blocks=3)
    output = residual_blocks(output, n_filters=128, strides=2, n_blocks=4)
    output = residual_blocks(output, n_filters=256, strides=2, n_blocks=6)
    output = residual_blocks(output, n_filters=512, strides=2, n_blocks=3)

    # fc --------------------- #
    output = keras.layers.GlobalAveragePooling2D()(output)
    output = keras.layers.Dense(1000)(output)

    model = keras.models.Model(inputs=input_tensor, outputs=output)
    model.summary()


# resnet_34_simple()
# resnet_50_simple()
# resnet_50_shortcut()
# resnet_50_bottle_neck_1()
# resnet_50_bottle_neck_2()
