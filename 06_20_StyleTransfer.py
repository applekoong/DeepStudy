# 케라스 창시자에 나오는 스타일 트랜스퍼 코드 정리

import keras
import numpy as np


def load_img(img_path, size):
    return keras.preprocessing.image.load_img(img_path, target_size=size)


def preprocess_image(img_path):
    img = load_img(img_path, (img_height, img_width))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.vgg16.preprocess_input(img)
    return img


def deprocess_image(x):
    # ImageNet의 평균 픽셀 값을 더합니다
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def content_loss(base, combination):
    return keras.backend.sum(keras.backend.square(combination - base))


def gram_matrix(x):
    features = keras.backend.batch_flatten(keras.backend.permute_dimensions(x, (2, 0, 1)))
    gram = keras.backend.dot(features, keras.backend.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return keras.backend.sum(keras.backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x):
    a = keras.backend.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = keras.backend.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    return keras.backend.sum(keras.backend.pow(a + b, 1.25))


def get_image_size(img_path):
    width, height = keras.preprocessing.image.load_img(img_path).size
    img_height = 400
    img_width = int(width * img_height / height)
    
    return img_height, img_width


target_path = './portrait.jpg'
style_path = './Village Street.jpg'

img_height, img_width = get_image_size(target_path)

target_image = keras.backend.constant(preprocess_image(target_path))
style_image = keras.backend.constant(preprocess_image(style_path))

# 생성된 이미지를 담을 플레이스홀더
combination_image = keras.backend.placeholder((1, img_height, img_width, 3))
input_tensor = keras.backend.concatenate([target_image,
                                          style_image,
                                          combination_image], axis=0)

# 세 이미지의 배치를 입력으로 받는 VGG 네트워크를 만듭니다.
# 이 모델은 사전 훈련된 ImageNet 가중치를 로드합니다
model = keras.applications.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# 콘텐츠와 스타일 손실에 사용할 층
content_layer = 'block5_conv2'
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# 손실 항목의 가중치 평균에 사용할 가중치
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

# 모든 손실 요소를 더해 하나의 스칼라 변수로 손실을 정의합니다
loss = keras.backend.variable(0.)
layer_features = outputs_dict[content_layer]

target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

loss += content_weight * content_loss(target_image_features, combination_features)

for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    
    loss += (style_weight / len(style_layers)) * sl
    
loss += total_variation_weight * total_variation_loss(combination_image)

# 손실에 대한 생성된 이미지의 그래디언트를 구합니다
# 현재 손실과 그래디언트의 값을 추출하는 케라스 Function 객체입니다
grads = keras.backend.gradients(loss, combination_image)[0]
fetch_loss_and_grads = keras.backend.function([combination_image], [loss, grads])

# ----------------------------------- # 

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

# ----------------------------------- # 

from scipy.optimize import fmin_l_bfgs_b

# 뉴럴 스타일 트랜스퍼의 손실을 최소화하기 위해 생성된 이미지에 대해 L-BFGS 최적화를 수행합니다
# 초기 값은 타깃 이미지입니다
# scipy.optimize.fmin_l_bfgs_b 함수가 벡터만 처리할 수 있기 때문에 이미지를 펼칩니다.
x = preprocess_image(target_path)
x = x.flatten()

evaluator = Evaluator()

for i in range(20):
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    print(i, min_val)

# ----------------------------------- # 

from matplotlib import pyplot as plt

plt.imshow(load_img(target_path, (img_height, img_width)))

plt.figure()
plt.imshow(load_img(style_path, (img_height, img_width)))

plt.figure()
plt.imshow(img)

plt.show()
