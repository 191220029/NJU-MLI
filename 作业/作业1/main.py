# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time

'''transfer class name to number'''


def class_to_int(s):
    class_name = {'CYT': 0, 'NUC': 1, 'MIT': 2, 'ME3': 3, 'ME2': 4,
                  'ME1': 5, 'EXC': 6, 'VAC': 7, 'POX': 8, 'ERL': 9}
    return class_name[s]


# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
    # url = "./data/yeast.data"
    rawData = pandas.read_csv(url, delim_whitespace=True, encoding="UTF-8",
                              names=['Sequence Name', 'mcg', 'gvh', 'alm', 'mit', 'erl', 'pox',
                                     'vac', 'nuc', 'class'], header=None, converters={9: class_to_int})
    print(rawData)
    # Initiate parameters
    cols = rawData.shape[1]
    # print(cols)
    X = rawData.iloc[:, 1:cols - 1]
    Y = rawData.iloc[:, cols - 1:cols]
    X = numpy.array(X)
    Y = numpy.array(Y)
    Y = Y.flatten()
    # print(X)
    # print(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    # print(X_train.shape, Y_train.shape)

    base_model = LogisticRegression(penalty='l2', solver='newton-cg', dual=False, class_weight='balanced',
                                    warm_start=False)
    model_ovr = OneVsRestClassifier(base_model)
    cur_time = time.time()
    model_ovr.fit(X_train, Y_train)
    print("ovr模型训练时间：%.3f s" % (time.time() - cur_time))
    print("ovr训练集的准确率：%.3f" % model_ovr.score(X_train, Y_train))
    print("ovr测试集的准确率：%.3f" % model_ovr.score(X_test, Y_test))
    Y_hat = model_ovr.predict(X_test)
    accuracy = metrics.accuracy_score(Y_test, Y_hat)
    print("ovr模型正确率：%.3f" % accuracy)
    print()

    model_ovo = OneVsOneClassifier(base_model)
    cur_time = time.time()
    model_ovo.fit(X_train, Y_train)
    print("ovo模型训练时间：%.3f s" % (time.time() - cur_time))
    print("ovo训练集的准确率：%.3f" % model_ovo.score(X_train, Y_train))
    print("ovo测试集的准确率：%.3f" % model_ovo.score(X_test, Y_test))
    Y_hat = model_ovo.predict(X_test)
    accuracy = metrics.accuracy_score(Y_test, Y_hat)
    print("ovo模型正确率：%.3f" % accuracy)
    print()

    model_multinomial = LogisticRegression(penalty='l2', solver='newton-cg', dual=False, class_weight='balanced',
                                           multi_class='multinomial', warm_start=False)
    cur_time = time.time()
    model_multinomial.fit(X_train, Y_train)
    print("multinomial模型训练时间：%.3f s" % (time.time() - cur_time))
    print("multinomial训练集的准确率：%.3f" % model_multinomial.score(X_train, Y_train))
    print("multinomial测试集的准确率：%.3f" % model_multinomial.score(X_test, Y_test))

    Y_hat = model_multinomial.predict(X_test)
    accuracy = metrics.accuracy_score(Y_test, Y_hat)  # 错误率，也就是np.average(y_test==y_hat)
    print("multinomial模型正确率：%.3f" % accuracy)
    print()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
