# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import spectral
import matplotlib.pyplot as plt
from scipy.io import loadmat
import logging
logger = logging.getLogger("HIC_Process")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.DEBUG)
class HIC(object):
    def __init__(self):

        # 真实地物图像
        self.input_image_path = 'E:\DataSets\matlabel\PaviaU.mat'
        self.input_image_path_class = 'paviaU' # mat文件的字段

        # 地物标签
        self.output_image_path = 'E:\DataSets\matlabel\PaviaU_gt.mat'
        self.output_image_path_class = 'paviaU_gt'  # mat文件字段

    # 预览高光谱图像和分类颜色标记
    def show_mat_image(self, view):
        """
        :param view:  str: view,    ground_truth,   overlay
        :return:  image: 一层地物图像，    地物标签，   覆盖标签的地物
        """

        input_image = loadmat(self.input_image_path)[self.input_image_path_class]  # 真实的图像
        output_image = loadmat(self.output_image_path)[self.output_image_path_class]   # 标签类别

        if view == 'view':
            # 查看真实地物的其中一层图像
            view = spectral.imshow(input_image, (30, 19, 9), classes=output_image)
            plt.show(view)

        elif view == 'ground_truth':
            # 查看真实地物的标签
            ground_truth = spectral.imshow(classes=output_image.astype(int), figsize=(9, 9))
            plt.show(ground_truth)
        elif view == 'overlay':
            # 将地物的标签覆盖到真实地物图像上
            view = spectral.imshow(input_image, (30, 19, 9), classes=output_image)
            view.set_display_mode('overlay')
            view.class_alpha = 0.5
            plt.show(view)

    # 将mat格式的图像文件转换成csv文件，方便后续处理
    def matImage_to_csv(self, csvImage_outputPath):
        """
        :param csvImage_outputPath:  转换后文件保存路径
        :return:
        """
        input_image = loadmat(self.input_image_path)[self.input_image_path_class]  # 真实的图像
        output_image = loadmat(self.output_image_path)[self.output_image_path_class]  # 标签类别
        logger.info("图像尺寸和张数：{}".format(input_image.shape))

        # 统计每类样本所含个数
        dict_k = {}
        for i in range(output_image.shape[0]):
            for j in range(output_image.shape[1]):
                if output_image[i][j] in [m for m in range(1, 17)]:  # 判断lable的范围在给定列表内
                    # if output_image[i][j] in [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13]:
                    if output_image[i][j] not in dict_k:
                        dict_k[output_image[i][j]] = 0
                    dict_k[output_image[i][j]] += 1

        # 将每一张图的像素所有点放在一列，像素点即样本数，图片个数为特征数
        need_label = np.zeros([output_image.shape[0], output_image.shape[1]])
        new_datawithlabel_list = []
        for i in range(output_image.shape[0]):
            for j in range(output_image.shape[1]):
                if output_image[i][j] != 0:
                    need_label[i][j] = output_image[i][j]

        for i in range(output_image.shape[0]):
            for j in range(output_image.shape[1]):
                if need_label[i][j] != 0:
                    c2l = list(input_image[i][j])
                    c2l.append(need_label[i][j])
                    new_datawithlabel_list.append(c2l)

        new_datawithlabel_array = np.array(new_datawithlabel_list)

        # 标准化数据并储存
        from sklearn import preprocessing
        data_D = preprocessing.StandardScaler().fit_transform(new_datawithlabel_array[:, :-1])
        data_L = new_datawithlabel_array[:, -1]

        new = np.column_stack((data_D, data_L))
        new_ = pd.DataFrame(new)
        new_.to_csv(csvImage_outputPath + '\out.csv', header=False, index=False)

        logger.info("Mat文件图像转换完成！路径保存至：{}".format(csvImage_outputPath))

    def predictAndshow(self, csvData_path):
        """
        :param csvData_path: CSV文件路径
        :return: 20 % 测试集的分类准确率， 所有样本点预测成像效果图
        """
        from sklearn.model_selection import train_test_split
        data = np.loadtxt(csvData_path, dtype=np.float, delimiter=",")
        data_D = data[:, :-1]
        data_L = data[:, -1]

        # 切割数据集
        X_train, X_test, y_train, y_test = train_test_split(data_D, data_L, test_size=0.2)

        # 选择分类模型
        """
        这里训练的时候选择训练集进行训练
        测试的时候用整个数据集来进行测试
        """
        from sklearn import svm
        from sklearn.metrics import accuracy_score
        clf = svm.SVC()
        clf.fit(X_train, y_train)
        pre_y = clf.predict(data_D)
        logger.info("The SVM Accuracy:{}".format(accuracy_score(data_L, pre_y) * 100))

        # 对所有样本点进行预测，然后将预测结果覆盖到类标签上，查看图像上的预测效果
        output_image = loadmat(self.output_image_path)[self.output_image_path_class]

        new_show = np.zeros((output_image.shape[0], output_image.shape[1]))
        k = 0
        for i in range(output_image.shape[0]):
            for j in range(output_image.shape[1]):
                if output_image[i][j] != 0:
                    new_show[i][j] = pre_y[k]
                    k += 1

        ground_predict = spectral.imshow(classes=new_show.astype(int), figsize=(9, 9))
        plt.show(ground_predict)



if __name__ == '__main__':
    hic = HIC()

    # 查看地物图像
    """
    参数：view  地物其中一层图像
          ground_truth  真实地物标签
          overlay  标签覆盖地物上的图像
    """
    # hic.show_mat_image('view')

    # 将mat格式的图像文件转换成csv格式
    """
    mat格式图像下载地址：
    http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes?tdsourcetag=s_pctim_aiomsg
    """
    # hic.matImage_to_csv('E:\DataSets\matlabel')

    # 分类预测，并将预测标签覆盖的地物上展示
    hic.predictAndshow('E:\DataSets\matlabel\out.csv')


