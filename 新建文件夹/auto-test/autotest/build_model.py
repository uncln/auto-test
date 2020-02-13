import tensorflow as tf
from tensorflow import keras
import numpy as np
# import matplotlib.pyplot as plt




fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() #加载数据集

class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat',
             'Sandal','Shirt','Sneaker','Bag','Ankle boot'] #图片的分类

# print("The shape of train_images is ",train_images.shape)
# print("The shape of train_labels is ",train_labels.shape)
# print("The shape of test_images is ",test_images.shape)
# print("The length of test_labels is ",len(test_labels))  #打印出图片的部分信息，大小，分辨率什么的，（没什么意义）

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.gca().grid(False)
# plt.show() #可视化数据，注意此处引入了限制外的包


train_images=train_images/255.0
test_images=test_images/255.0   #归一化
#
# # plt.figure(figsize=(10,10))
# # for i in range(25):
# #     plt.subplot(5,5,i+1)
# #     plt.xticks([])
# #     plt.yticks([])
# #     plt.grid('off')
# #     plt.imshow(train_images[i],cmap=plt.cm.binary)
# #     plt.xlabel(class_names[train_labels[i]])
# # plt.show()  #可视化归一化之后的代码
#
#
#接下来开始构建我们自己的模型！！敲黑板重点

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])  #模型的配置层 ，看得懂个p

model.compile(optimizer='adam',#AdamOptimizer
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#编译该模型

model.fit(train_images,train_labels,epochs=71) #训练该模型

test_loss,test_acc=model.evaluate(test_images,test_labels)
print('Test Acc:',test_acc)#评估准确率


model.save('model.h5')            #模型的保存

#重新创建完全相同的模型，包括其权重和优化程序
new_model = keras.models.load_model('model.h5')

predictions=new_model.predict(test_images)
print("The first picture's prediction is:{},so the result is:{}".format(predictions[1],np.argmax(predictions[1])))
print("The first picture is ",test_labels[1])

#模型的预测
