#!coding:utf8
import numpy as np
import matplotlib.pyplot as plt
#主要学习如下知识点：
#读取minst的csv文件，文件中每一行描述了一张28,28的图片，且包含了这张图片显示的数字0~9
#读数据进行转换，得到图片array，使用matplotlib显示图片的方法
#将图片的数字值 ，转换为one hot list


def prepare_image(img_str):
    """
    :param img_str:  image represented by str
    :return:  array of one image, one hot output list
    """
    img_list = img_str.split(",")
    label = int(img_list[0])
    output = np.zeros(10) + 0.01
    output[label] = 0.99
    data_array = np.asfarray(img_list[1:]).reshape((28,28))
    scaled_image = data_array/255.0 * 0.99 + 0.01
    # plt.imshow(scaled_image,cmap="Greys")
    # plt.show()
    return scaled_image, output


if __name__ =="__main__":
    fd = open("./mnist_dataset/mnist_train_100.csv")
    data = fd.readlines()
    img_array, output = prepare_image(data[0])
    print(img_array)
    print (output)
    # print (img_array.reshape(1,28*28).T.shape)





