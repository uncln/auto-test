def average(image):
    (width, height) = image.shape
    ave = 0.0
    for i in range(width):
        for j in range(height):
            ave += image[i][j]
    return ave / 784
#平均灰度 μx 即亮度对比函数

def deviation(img, ave):
    (width, height) = img.shape
    dev = 0.0
    for i in range(width):
        for j in range(height):
            dev += (img[i][j] - ave) * (img[i][j] - ave)
    return (dev / 783) ** 0.5
#对比度对比函数


def assit_function(img1, ave1, img2, ave2):
    (width, height) = img1.shape
    dev = 0.0
    for i in range(width):
        for j in range(height):
            dev += (img1[i][j] - ave1) * (img2[i][j] - ave2)
    return dev / (28 * 28 - 1)
#计算结构对比函数

def SSIM(image1, image2):
    K1 = 0.01
    K2 = 0.03
    L = 255
    C1 = (K1 / L) * (K1 / L)
    C2 = (K2 / L) * (K2 / L)
    C3 = C2 / 2
    ave1 = average(image1)
    ave2 = average(image2)

    L = (2 * ave1 * ave2 + C1) / (ave1 * ave1 + ave2 * ave2 + C1) #图像灰度级数

    dev1 = deviation(image1, ave1)
    dev2 = deviation(image2, ave2)

    C = (2 * dev1 * dev2 + C2) / (dev1 * dev1 + dev2 * dev2 + C2)#对比度对比函数

    deviation12 = assit_function(image1, ave1, image2, ave2)

    S = (deviation12 + C3) / (dev1 * dev2 + C3)#结构对比函数

    return L * C * S
