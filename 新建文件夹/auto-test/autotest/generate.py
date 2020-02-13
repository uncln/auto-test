import random
from tensorflow import keras
import tensorflow as tf
from ssim import SSIM
import copy
import time
import numpy as np

ccc=0

def generate(images,shape):
    isstandard=1
    (count,w,h,n)=shape
    images = images.reshape(count,w,h)
    for i in images[0]:
        for j in i:
            if(j>1):
               isstandard=0
    if(isstandard==0):
        #print("did")
        images=images/255.0
    generate_images=[]
    j=0
    while j<len(images):
        temp = attack(images[j])
        generate_images.append(temp)
        print("get"+str(j))
        j=j+1
    generate_images = np.array(generate_images)
    generate_images = generate_images.reshape(count,w,h,n)

    live = generate_images.reshape(10000,28,28,1)
    np.save('1',live)
    print(time.time())
    return generate_images


def attack(image):
    temps=[]
    start = time.time()
    for count in range(20):#可修改的最少遍历次数
        succ=0
        new_image = random_attack(image, 500) #可修改的遍历深度
        if (new_image is not None):
            # 攻击成功
            succ=1
        else:
            print("fail")
            ran = random.randint(0,10000)
            global ccc
            ccc+=1
            return train_images[ran]/255.0
            # 失败 条件为500*20次无法生成相似度>0.75的标签不同的新图片 参数可修改
        temps.append((new_image,SSIM(new_image, image)))
        if(SSIM(new_image, image) > 0.9):
            break
    index = 0
    max_ssim = 0
    for i in range(len(temps)):
        ssim = temps[i][1]
        if ssim > max_ssim:
            index = i
            max_ssim = ssim
    #print(time.time()-start)
    #print(max_ssim)
    res = temps[index]
    return res[0]


def random_line_attack(image,r):
    now_image = copy.deepcopy(image)
    (width, height) = now_image.shape
    x = random.randint(0, width - 1)
    y = random.randint(0, height - 1)
    color = random.randint(0, 255)
    color = color/255.0
    for times in range(r):
        if y + times < width:
            #color = now_image[y+times][y+times]
            for i in range(height):
                now_image[i][y + times] = color
        if x + times < height:
            #color = now_image[x+times][x+times]
            for i in range(width):
                now_image[x + times][i] = color
    return now_image


def random_attack(image, max_time):
    label = predict(image)
    loop=1
    new_image = random_line_attack(image, 1)
    i = 0
    while (predict(new_image) == label or SSIM(new_image, image) < 0.75) and i < max_time:#结构相似度可调整
        i += 1
        new_image = random_line_attack(image, r=loop + 1)
    if i < max_time:
        return new_image
    return None

def predict(images):
    image = images.reshape(1,28,28)
    predictions = new_model.predict(image)
    return np.argmax(predictions[0])



#new_model = keras.models.load_model('model.h5') 舍弃的加载模型方式

saved_model_path = "models.h5".format(int(time.time()))

#tf.keras.experimental.export_saved_model(new_model, saved_model_path) 新的模型存储方式

new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)

#加载模型

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

test_images = test_images

print(len(test_images))

shape=(10000,28,28,1)

real_tests = generate(test_images,shape) #生成测试集对应的对抗样本集合