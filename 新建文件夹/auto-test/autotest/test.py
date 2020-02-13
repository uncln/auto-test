import tensorflow as tf
import time
from tensorflow import keras
import  numpy as np

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


saved_model_path = "models.h5".format(int(time.time()))

#tf.keras.experimental.export_saved_model(new_model, saved_model_path) 新的模型存储方式

new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)

new_model.compile(optimizer='adam',#AdamOptimizer
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

t=np.load("attack_data.npy")

t=t.reshape(10000,28,28)

test_loss,test_acc=new_model.evaluate(t,test_labels)

print('Test Acc:',test_acc)#评估准确率
