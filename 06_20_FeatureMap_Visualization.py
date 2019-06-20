# CNN 아키텍처에 포함된 컨볼루션 레이어에서 생성되는 피처맵 시각화
# 학습이 끝난 모델 파일 필수(Model/cats_and_dogs_small_2.h5)
# 피처맵에 반영될 이미지 필수(cats_and_dogs/small/test/cats/cat.1503.jpg)


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def show_image(img_path, target_size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_tensor = tf.keras.preprocessing.image.img_to_array(img)

    # print(img_tensor[0, 0])                         # [31.  2.  6.]
    # print(np.min(img_tensor), np.max(img_tensor))   # 0.0 255.0
    # print(img_tensor.dtype, img_tensor.shape)       # float32 (150, 150, 3)

    # 스케일링하지 않으면 이미지 출력 안됨
    img_tensor /= 255

    plt.imshow(img_tensor)
    plt.show()


def load_image(img_path, target_size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_tensor = tf.keras.preprocessing.image.img_to_array(img)

    return img_tensor[np.newaxis] / 255     # (1, 150, 150, 3)


def show_first_activation_map(model_path, img_path):
    model = tf.keras.models.load_model(model_path)
    # model.summary()

    first_output = model.layers[0].output
    print(first_output.shape, first_output.dtype)   # (?, 148, 148, 32) <dtype: 'float32'>

    # 1개의 출력을 갖는 새로운 모델 생성
    sample_model = tf.keras.models.Model(inputs=model.input, outputs=first_output)

    img_tensor = load_image(img_path, (model.input.shape[1], model.input.shape[2]))

    print(model.input.shape)            # (?, 150, 150, 3)
    print(img_tensor.shape)             # (1, 150, 150, 3)

    first_activation = sample_model.predict(img_tensor)

    print(first_activation.shape)       # (1, 148, 148, 32)
    print(first_activation[0, 0, 0])    # [0.00675746 0. 0.02397328 0.03818807 0. ...]

    # 19번째 활성 맵 출력
    plt.matshow(first_activation[0, :, :, 19], cmap='gray')     # viridis
    plt.show()


def show_activation_maps(layer, title, n_cols=16):
    size, n_features = layer.shape[1], layer.shape[-1]
    assert n_features % n_cols == 0

    n_rows = n_features // n_cols
    big_image = np.zeros((n_rows*size, n_cols*size), dtype=np.float32)

    for row in range(n_rows):
        for col in range(n_cols):
            # 특정 부분에 값들이 편중되어 있는 상태
            # 피처에 대해 반복하기 때문에 channel 변수는 피처맵 1개를 가리킨다.
            channel = layer[0, :, :, row * n_cols + col]      # shape : (size, size)

            # 특성이 잘 보이도록 0~255 사이로 재구성. 범위를 벗어나는 일부 값들은 클리핑.
            # 대략적으로 정규 분포에 따르면 95% 정도의 데이터 포함
            channel -= channel.mean()
            channel /= channel.std()
            channel *= 64
            channel += 128
            channel = np.clip(channel, 0, 255).astype('uint8')

            big_image[row*size:(row+1)*size, col*size:(col+1)*size] = channel

    # 그래프에서는 행열(n_rows, n_cols) 순서가 xy로 바뀐다.
    plt.figure(figsize=(n_cols, n_rows))

    plt.xticks(np.arange(n_cols) * size)
    plt.yticks(np.arange(n_rows) * size)
    plt.title(title)
    # plt.tight_layout()
    plt.imshow(big_image)           # cmap='gray'


def predict_and_get_outputs(model_path, img_path):
    model = tf.keras.models.load_model(model_path)

    # (conv, pool) 레이어만 출력. flatten 레이어 이전까지.
    layer_outputs = [layer.output for layer in model.layers[:8]]
    layer_names = [layer.name for layer in model.layers[:8]]

    print([str(output.shape) for output in layer_outputs])
    # ['(?, 148, 148, 32)', '(?, 74, 74, 32)', '(?, 72, 72, 64)', '(?, 36, 36, 64)',
    #  '(?, 34, 34, 128)', '(?, 17, 17, 128)', '(?, 15, 15, 128)', '(?, 7, 7, 128)']

    # 마지막 레이어에 해당하는 7번째 레이어의 출력만 전달해도 될 것 같지만, 7번째 결과만 나옴.
    # train 연산에 loss가 포함되어 있지만, loss를 명시하지 않으면 결과를 얻지 못하는 것과 같다.
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

    input_shape = (model.input.shape[1], model.input.shape[2])      # (150, 150)
    img_tensor = load_image(img_path, target_size=input_shape)

    layer_outputs = activation_model.predict(img_tensor)
    return layer_outputs, layer_names


model_path = 'Model/cats_and_dogs_small_2.h5'
img_path = 'cats_and_dogs/small/test/cats/cat.1503.jpg'
# img_path = 'cats_and_dogs/small/test/cats/cat.1700.jpg'

# --------------------------------------------- #

layer_outputs, layer_names = predict_and_get_outputs(model_path, img_path)

for layer, name in zip(layer_outputs, layer_names):
    show_activation_maps(layer, name)

plt.show()
