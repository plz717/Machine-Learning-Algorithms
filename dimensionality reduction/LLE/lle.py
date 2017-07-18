import numpy as np
from isomap import Isomap
from scipy import linalg as LA
from pca import load_data
int = long


class LLE(object):
    """Input:
    Data: array of samples with shape (num_samples,sample_dimension).
    k: how many neareset neighbors to found in the k-nearest algorithm.
    d_: the low dimension expected.
    """

    def __init__(self, Data, k, d_):
        self.d_ = d_
        self.k = k
        self.D = Data
        self.sample_num = Data.shape[0]
        self.W = np.zeros((self.sample_num, self.sample_num))

    def calcu_W(self):
        for i in range(self.sample_num):
            """obtain the k-nearest neighbors of every point"""
            print("i is:{}".format(i))
            x_instance = Isomap(self.D, self.k, self.d_)
            index_sorted, _ = x_instance.search_k_mins_of_a_point(self.D[i], i)
            # the index of the k nearest neighbors of point xi
            k_mins_index = index_sorted[-self.d_:]
            print("k_mins_index is:", k_mins_index)
            for j in k_mins_index:
                if j != i:
                    xi = self.D[i]
                    xj = self.D[j]
                    # calculate the upper part
                    sum_upper = 0
                    for k in k_mins_index:
                        if k != i:
                            xk = self.D[k]
                            # C_jk is a value
                            C_jk = np.dot((xi - xj).astype(float),
                                          (xi - xk).astype(float))
                            print("C_jk is:{}".format(C_jk))
                            if C_jk == 0:
                                print("these two vectors are orthogonal")
                            sum_upper += (1 / float(C_jk))
                            print("sum_upper is:{}".format(sum_upper))
                    # calculate the lower part
                    sum_down = 0
                    for ll in k_mins_index:
                        for s in k_mins_index:
                            if (s != i) and (ll != i):
                                xl = self.D[ll]
                                xs = self.D[s]
                                C_ls = np.dot(
                                    (xi - xl).astype(float), (xi - xs).astype(float))
                                print("C_ls is:{}".format(C_ls))
                                sum_down += (1 / float(C_ls))
                    print("sum_upper is:{},sum_upper is:{}".format(
                        sum_upper, sum_down))
                    self.W[i][j] = float(sum_upper) / float(sum_down)

        return self.W

    def calcul_M(self):
        idty_array = np.eye(self.sample_num)
        M = np.dot((idty_array - self.W).T.astype(float),
                   (idty_array - self.W).astype(float))
        print("M shape is:{}".format(M.shape))
        return M

    def calcu_Z_transpose(self):
        """ eigen_vectors according to d_ minimal eigen_values of M form into the Z.T
        namely the samples in the new low-dimension space.
        """
        M = self.calcul_M()
        e_vals, e_vecs = LA.eig(M)
        print("e_vals is:{}".format(e_vals))
        print("e_vecs shape is:{}".format(e_vecs.shape))
        eigen_vector_array = e_vecs[:, -self.d_:]
        eigen_vector_array = eigen_vector_array[
            :, ::-1]  # flipping on the cols
        # print("eigen_vector_array shape is:{}".format(eigen_vector_array.shape))
        return eigen_vector_array


if __name__ == '__main__':
    train_faces = load_data('./dataset', 1, 10)
    d_ = input("input the dimensions to be reducted:")
    k = input("how many neighbors does a point have:")
    instance = LLE(train_faces, k, d_)
    w = instance.calcu_W()
    instance.calcul_M()
    eigen_vector_array = instance.calcu_Z_transpose()
