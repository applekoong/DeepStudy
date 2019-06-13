# talos 사용법
# 1. 순서
#    Scan -> Predict -> Evaluate -> Deploy
#    Restore -> Reporting
# 2. 주의사항
#    [gpu]
#    tf.keras 사용하면 GlorotUniform 에러
#    keras 사용하면 성공
#    [tpu]
#    colab에서 tpu 사용하면 훨씬 더 빨리 처리 가능
#    tf.keras 사용하면 Scan 객체는 만들어지는데 GlorotUniform 에러
#    keras 사용하면 괜찮은데 Scan 함수의 model 매개변수에서 타입 에러 발생
# 3.
#    텐서플로 2.0에서 contrib 모듈 제거됨
#    eager, tpu, lite는 core로 이동
#    contrib에 있는 tpu 기능을 사용하기 때문에 외부 keras 모듈과 안 맞을 수 있음
#    2.0이 되면 tf.keras와 연동할 수 있을 거라 추측
#    그 동안은 tpu 사용 금지. keras 모듈 사용

import talos as ta
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

results = ta.Reporting('breast_cancer_1.csv')
print(results)          # <talos.commands.reporting.Reporting object at 0x103418630>
print(type(results))    # <class 'talos.commands.reporting.Reporting'>

print(type(results.data))                   # <class 'pandas.core.frame.DataFrame'>
print(results.data.columns)
# first_neuron부터 직접 추가한 옵션. fmeasure_acc까지는 자동으로 추가된 옵션
# ['round_epochs', 'val_loss', 'val_acc', 'val_fmeasure_acc', 'loss', 'acc', 'fmeasure_acc',
#  'first_neuron', 'hidden_layers', 'batch_size', '*epochs', 'dropout', 'kernel_initializer', 'optimizer', 'losses', 'activation', 'last_activation']

print(results.high('val_acc'))              # 0.9824561309396176
print(results.high('val_fmeasure_acc'))     # 0.9762043566034552
print(results.high('fmeasure_acc'))         # 0.9931475490181888
print(results.high('acc'))                  # 0.9924623118573098
print(results.data)
#     round_epochs  val_loss   val_acc  val_fmeasure_acc      loss       acc  fmeasure_acc  first_neuron  hidden_layers  batch_size  epochs  dropout kernel_initializer                         optimizer                                            losses                         activation last_activation
# 0            100  0.066695  0.970760          0.962709  0.047446  0.989950      0.989053             9              1          30     100        0             normal   <class 'keras.optimizers.Adam'>  <function binary_crossentropy at 0x7fd450a982f0>   <function elu at 0x7fd450a3df28>         sigmoid
# 1            100  0.065699  0.970760          0.961352  0.045587  0.989950      0.989254             9              2          30     100        0            uniform   <class 'keras.optimizers.Adam'>  <function binary_crossentropy at 0x7fd450a982f0>  <function relu at 0x7fd450a4b1e0>         sigmoid

# val_loss 기준으로 정렬
print(results.table('val_loss'))

# results.plot_line('val_acc')
# results.plot_box('val_acc')
# results.plot_corr('val_acc')
# plt.show()

print('-' * 50)

# predict와 evaluate 함수에는 Reporing 객체 전달하면 saved_models 에러 발생.
# Scan 객체 전달해야 함.
# https://nbviewer.jupyter.org/github/autonomio/talos/blob/master/examples/Hyperparameter%20Optimization%20with%20Keras%20for%20the%20Iris%20Prediction.ipynb

x, y = ta.datasets.breast_cancer()
x = preprocessing.StandardScaler().fit_transform(x)

p = ta.Predict(results)
model_id = results.best_params(metric='val_acc')
print(model_id)
# print(results.predict(x))
#
e = ta.Evaluate(results)
print(e.evaluate(x, y))
