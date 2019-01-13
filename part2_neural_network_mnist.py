import matplotlib.pyplot as plt
def prepare_image(img_str,isshow=False):
    """
    :param img_str:  image represented by str
    :return:  array of one image, one hot output list
    """
    img_list = img_str.split(",")
    label = int(img_list[0])
    # print (f"right digit:{label}")
    output = np.zeros(10) + 0.01
    output[label] = 0.99
    data_array = np.asfarray(img_list[1:]).reshape((28,28))
    scaled_image = data_array/255.0 * 0.99 + 0.01
    if isshow:
        plt.imshow(scaled_image,cmap="Greys")
        plt.show()
    scaled_image = scaled_image.ravel()
    return scaled_image, output

import scipy.special
import numpy as np
class NeuralNetwork:
    def __init__(self,input_node_count,hidden_node_count,output_node_count,learn_rate):
        self.input_node_count = input_node_count
        self.hidden_node_count = hidden_node_count
        self.output_node_count = output_node_count

        self.activate = scipy.special.expit
        self.learn_rate = learn_rate

        # self.wih
        # self.who
        #第二个参数是方差，方差越小，数据越集中，靠的越近
        #这需要保证：每个计算结点，w的个数越多，每个w的取值就要越小
        self.wih = np.random.normal(0.0,pow(self.input_node_count,-0.5),(self.hidden_node_count,self.input_node_count))
        self.who = np.random.normal(0.0,pow(self.hidden_node_count , -0.5),(self.output_node_count,self.hidden_node_count))

    def query(self,input_list):
        inputs = np.array(input_list,ndmin = 2) #转化为二维数组1xn
        inputs = inputs.T
        hidden_output = np.dot(self.wih,inputs)
        hidden_output = self.activate(hidden_output)

        output = np.dot(self.who,hidden_output)
        output = self.activate(output)
        return output

    def train(self,input_list,target_list):
        inputs = np.array(input_list,ndmin = 2) #转化为二维数组1xn
        inputs = inputs.T
        target = np.array(target_list,ndmin = 2)
        target = target.T

        hidden_output = np.dot(self.wih,inputs)
        hidden_output = self.activate(hidden_output)

        output = np.dot(self.who,hidden_output)
        output = self.activate(output)

        # w = w- lr*dw
        #loss_function =(a-y)**2
        #da = (a-y)
        da = (output - target) # 10x 1
        dz = da * output * (1.0-output)
        hidden_output_error = np.dot( self.who.T, dz)

        #10x1 100x1
        dwho = np.dot(dz,hidden_output.T)

        dwih =np.dot(hidden_output*(1.0-hidden_output)*hidden_output_error,inputs.T)
        self.who -= self.learn_rate *dwho
        self.wih -= self.learn_rate *dwih

def show_norm_fig(peak_point_count):
    #采集的点数越多，就代表w的个数越多，所以，需要每个w的权值更小，避免进入梯度无法快速下载的区域
    norm_data = np.random.normal(0, pow(peak_point_count,-0.5), (1, peak_point_count))
    plt.hist(norm_data[0],100)
    plt.show()


if __name__ =="__main__":
    fd = open("./mnist_dataset/mnist_train_100.csv")
    data = fd.readlines()
    fd.close()
    input_feature = 28*28
    hidden_count = 200
    output_count = 10
    learn_rate = 0.1
    nn = NeuralNetwork(input_feature,hidden_count,output_count,learn_rate)
    epochs = 40
    count = 1
    for i in range(epochs):
        for img_str in data:
            img_list, target = prepare_image(img_str)
            nn.train(img_list,target)
    #test model
    test_data_file = open("mnist_dataset/mnist_test_10.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    result = []
    for img_str in test_data_list:
        img_list, target = prepare_image(img_str,True)
        output = nn.query(img_list)
        print (output)
        label = np.argmax(output)  #通过model得到的数字
        correct_label = np.argmax(target) #这是正确的数字
        print(u"正确数字是：",correct_label)
        print(u"模型得到的数字是:",label)
        if correct_label == label:
            result.append(1)
        else:
            result.append(0)

    print (u"正确率:")
    print (sum(result)/float(len(result)))