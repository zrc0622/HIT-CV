from sift import SIFT
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import random

random.seed(10)

# 构建视觉词袋
def build_vocabulary(data, k, max_iters=100):
    # 迭代最大次数为100
    # 随机初始化聚类中心
    centroids = data[np.random.choice(len(data), k, replace=False)]
    
    for _ in range(max_iters):
        # 将每个数据点分配到最近的聚类中心
        distances = np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # 更新聚类中心
        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)

    return centroids

# 视觉词频统计
def calculate_word_frequencies(descriptors, vocabulary):
    distances = np.linalg.norm(descriptors[:, np.newaxis, :] - vocabulary, axis=2)
    closest_words = np.argmin(distances, axis=1)
    word_frequencies = np.bincount(closest_words, minlength=len(vocabulary))
    total_sum = sum(word_frequencies)
    frequencies = [element / total_sum for element in word_frequencies]
    return frequencies

# 训练分类器SVM
def train_classifier(X, y):
    scaler = StandardScaler() # 每个特征的均值为0，标准差为1
    clf = make_pipeline(scaler, SVC(kernel='linear')) # clf包括了scaler放缩和svm预测两个步骤
    clf.fit(X, y)
    return clf

if __name__ == '__main__':
    train_folder = './data/training/'
    test_folder = './data/testing/'
    my_train_folder = './data/my_training/'
    my_test_folder = './data/my_testing/'

    descriptors_list = [] # 描述符列表, shape为（图片数量 描述符数量 描述符维度）
    labels_list = [] # 标签列表, shape为（图片数量）

    # 提取描述符
    for root, dirs, files in os.walk(train_folder):
        # 随机抽取十分之一的文件
        sampled_files = random.sample(files, len(files) // 1000)
        for file in tqdm(sampled_files, desc="Processing Images", unit=" image"):
            if file.endswith(".jpg"):
                img_path = os.path.join(root, file)
                label = int(file[0])  # 从文件名中提取标签

                origin_img = plt.imread(img_path)
                if len(origin_img.shape) == 3:
                    img = origin_img.mean(axis=-1)
                else:
                    img = origin_img
                keyPoints, descriptors = SIFT(img)

                descriptors_list.append(np.array(descriptors))
                labels_list.append(label)

    unique_labels = np.unique(labels_list)
    combined_descriptors = [np.concatenate([desc for desc, label in zip(descriptors_list, labels_list) if label == unique_label], axis=0) for unique_label in unique_labels] # 按类别组合
    classes_num = 5 # 类别数量
    vocabulary_size = 40 # 词袋数量

    # 构建词袋
    all_descriptors = np.vstack(descriptors_list)

    total_vocabulary = np.empty((0, 128))

    for descriptors_small_list in combined_descriptors:
        print(descriptors_small_list.shape)
        vocabulary = build_vocabulary(descriptors_small_list, vocabulary_size)
        total_vocabulary = np.vstack([total_vocabulary, vocabulary])

    vocabulary = total_vocabulary

    # 每张图片的视觉词频统计
    X_train = []

    for descriptors in descriptors_list:
        word_frequencies = calculate_word_frequencies(descriptors, vocabulary)
        X_train.append(word_frequencies)

    X_train = np.array(X_train)
    Y_train = np.array(labels_list)

    # 训练分类器
    clf = train_classifier(X_train, Y_train)

    # 提取测试集描述符
    test_descriptors_list = [] # 描述符列表
    test_labels_list = [] # 标签列表

    # 提取描述符
    for root, dirs, files in os.walk(test_folder):
        # 随机抽取十分之一的文件
        sampled_files = random.sample(files, len(files) // 200)

        for file in tqdm(sampled_files, desc="Processing Images", unit=" image"):
            if file.endswith(".jpg"):
                img_path = os.path.join(root, file)
                label = int(file[0])  # 从文件名中提取标签

                origin_img = plt.imread(img_path)
                if len(origin_img.shape) == 3:
                    img = origin_img.mean(axis=-1)
                else:
                    img = origin_img
                keyPoints, descriptors = SIFT(img)

                test_descriptors_list.append(np.array(descriptors))
                test_labels_list.append(label)

    # 计算测试集的视觉单词频率
    X_test = []
    for descriptors in test_descriptors_list:
        # print(descriptors.shape)
        # print(vocabulary.shape)
        word_frequencies = calculate_word_frequencies(descriptors, vocabulary)  # 使用训练集的词典
        X_test.append(word_frequencies)

    X_test = np.array(X_test)
    Y_test = np.array(test_labels_list)

    # 统计识别准确率
    from sklearn.metrics import accuracy_score, classification_report

    def test_classifier(clf, X_test, y_test):
        # 预测测试集标签
        y_pred = clf.predict(X_test)

        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

        # 打印分类报告
        report = classification_report(y_test, y_pred)
        print('Classification Report:\n', report)

    # 使用已经训练好的分类器进行测试
    test_classifier(clf, X_train, Y_train)
    test_classifier(clf, X_test, Y_test)