# 0번째 레이어의 32개 필터를 채널별로 모두 출력

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def deprocess_image(weights):
    weights -= np.mean(weights)
    weights /= (np.std(weights) + 1e-5)
    weights *= 0.1

    weights += 0.5
    weights = np.clip(weights, 0, 1)

    weights *= 255
    weights = np.clip(weights, 0, 255).astype('uint8')

    return weights


def show_first_filters(model_path):
    model = tf.keras.models.load_model(model_path)
    # model.summary()

    # bias는 사용 안함
    weights, bias = model.layers[0].get_weights()
    weights = deprocess_image(weights)

    print(weights.shape)        # (3, 3, 3, 32)

    big_image = np.zeros((3 * 3 + 2, 3 * 32 + 31))
    for i in range(32):
        w = weights[:, :, :, i]

        pos = i * (3 + 1)
        big_image[ :3, pos:pos+3] = w[:, :, 0]
        big_image[4:7, pos:pos+3] = w[:, :, 1]
        big_image[8: , pos:pos+3] = w[:, :, 2]

    plt.matshow(big_image, cmap='gray')
    plt.show()


model_path = 'Model/cats_and_dogs_small_2.h5'
show_first_filters(model_path)
