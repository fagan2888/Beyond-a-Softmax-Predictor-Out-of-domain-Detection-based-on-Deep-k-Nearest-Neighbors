import numpy as np
import matplotlib.pyplot as plt

def MNIST_To_CIFAR_FORM(mnist_train_x, mnist_train_y,mnist_test_x, mnist_test_y):
    """
    Change the one-channel to RBG-channel on mnist_train_x and mnist_test_x
    Change the shape of mnist_train_y and mnist_test_y from (length) to (length,1)
    ---------------------------------------
    inputs:
    mnist_train_x, mnist_train_y,mnist_test_x, mnist_test_y which is all multi-dimension array
    It is recommended to use the following way to import the data
    ========================== codes ==========================
    mnist = keras.datasets.mnist
    (mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y)\
    = mnist.load_data()
    ========================== codes ==========================
    outputs:
    mnist_train_RGB_x, M_train_y, mnist_test_RGB_x, M_test_y 
    """
    from skimage import exposure
    import imutils
    B= []
    for i in range(len(mnist_train_x)):
        A = mnist_train_x[i]
        A = exposure.rescale_intensity(A, out_range=(0, 255))
        A = imutils.resize(A, width=32)
        B.append(A)
    B = np.array(B)

    mnist_train_RGB_x = np.repeat(B[:,:, :, np.newaxis], 3, axis=3)
    B= []
    for i in range(len(mnist_test_x)):
        A = mnist_test_x[i]
        A = exposure.rescale_intensity(A, out_range=(0, 255))
        A = imutils.resize(A, width=32)
        B.append(A)
    B = np.array(B)

    mnist_test_RGB_x = np.repeat(B[:,:, :, np.newaxis], 3, axis=3)
    M_train_y = np.array([[mnist_train_y[i]] for i in range(len(mnist_train_y))])
    M_test_y = np.array([[mnist_test_y[i]] for i in range(len(mnist_test_y))])
    return mnist_train_RGB_x, M_train_y, mnist_test_RGB_x, M_test_y

def find_idx(arr, target):
    ans = []
    for i in range(len(arr)):
        if arr[i] == target:
            ans.append(i)
    return ans
def get_submax(arr):
    arr = np.array(arr)
    MAX = np.max(arr)
    idx = find_idx(arr, MAX)
    arr_without_max = np.delete(arr,idx)
    return np.max(arr_without_max)
def find_statistics(Prob_Mat):
    Prob_diff = []
    MAX_Prob_Mat = []
    MAX_Prob_Mat_idx = []
    subMAX_Prob_Mat = []
    subMAX_Prob_Mat_idx = []
    for i in range(len(Prob_Mat)):
        MAX = np.max(Prob_Mat[i])
        MAX_idx = find_idx(Prob_Mat[i], MAX)[0]
        subMAX = get_submax(Prob_Mat[i])
        subMAX_idx = find_idx(Prob_Mat[i], subMAX)[0]
        prob_difference = MAX - subMAX
        Prob_diff.append(prob_difference)
        MAX_Prob_Mat.append(MAX)
        subMAX_Prob_Mat.append(subMAX)
        MAX_Prob_Mat_idx.append(MAX_idx)
        subMAX_Prob_Mat_idx.append(subMAX_idx)
    return Prob_diff,MAX_Prob_Mat,MAX_Prob_Mat_idx,subMAX_Prob_Mat,subMAX_Prob_Mat_idx
def separate_one_class(target_class_label, x_train, y_train, x_test, y_test):
    with_train_idx = find_idx(y_train, target_class_label)
    with_test_idx = find_idx(y_test, target_class_label)
    without_train_idx = list(set(range(len(y_train))).difference(set(with_train_idx)))
    without_test_idx = list(set(range(len(y_test))).difference(set(with_test_idx)))
    with_train = x_train[with_train_idx]
    with_test = x_test[with_test_idx]
    without_train = x_train[without_train_idx]
    without_test = x_test[without_test_idx]
    with_train_y = y_train[with_train_idx]
    with_test_y = y_test[with_test_idx]
    without_train_y = y_train[without_train_idx]
    without_test_y = y_test[without_test_idx]
    return with_train, with_train_y, without_train, without_train_y, with_test, with_test_y, without_test, without_test_y
def consistency_of_df(df, upper_bound, lower_bound):
    return len(df[df['arr_KNN_from_same_class_ratio']>upper_bound][df['arr_KNN_max_class_label'] == df['predicted_label']])/len(df[df['arr_KNN_from_same_class_ratio']>upper_bound]), len(df[df['arr_KNN_from_same_class_ratio']<lower_bound][df['arr_KNN_max_class_label'] == df['predicted_label']])/len(df[df['arr_KNN_from_same_class_ratio']<lower_bound])