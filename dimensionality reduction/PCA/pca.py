import numpy as np
from scipy import linalg as LA
import cv2
from PIL import Image


class PCA(object):
    def __init__(self, X):
        # substract the meanvalue of each column
        self.mean = np.mean(X, axis=0)
        self.X = X - self.mean
        self.conv = np.dot(X.T, X)  # (32*32) dimensions
        e_vals, e_vecs = LA.eig(self.conv)
        self.e_vecs_array = e_vecs

    def calcu_W(self, d_):
        self.W = self.e_vecs_array[:,:d_]
        return self.W

    def reduct(self):
        self.reduction = np.dot(self.X, self.W)
        return self.reduction

    def reconstruct(self):
        self.reconstruction = np.dot(self.reduction, self.W.T)
        return self.reconstruction


def load_data(dir, start_id, end_id):
    num = end_id - start_id + 1
    data = []
    for i in range(start_id, end_id + 1):
        img = cv2.imread(dir + '/' + str(i) + '.jpg',0)
        m, n= img.shape
        print("m is:{},n is:{}".format(m,n))
        data.append(img)

    data = np.array(data).reshape((num, m * n))
    return data


def array_to_img(dir, array, mean):
    m, n = array.shape
    img_side = int(np.sqrt(n))
    for i in range(m):
        img_array = array[i] + mean
        img_array = img_array.reshape((img_side, img_side))
        img = Image.fromarray(img_array)
        print("img type is:{},img mode is:{}".format(type(img),img.mode))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(dir+'/'+ str(i + 1) + '.jpg')
        #cv2.imwrite(dir + '/' + str(i + 1) + '.jpg', img)


if __name__ == '__main__':
    train_faces = load_data('./dataset', 1, 10)
    test_faces = load_data('./dataset', 11, 12)
    d_ = 100
    pca_train = PCA(train_faces)
    pca_W=pca_train.calcu_W(d_)
    pca_reduction = pca_train.reduct()
    pca_reconstruction = pca_train.reconstruct()
    array_to_img('./dataset/reconstruction', pca_reconstruction, pca_train.mean)
