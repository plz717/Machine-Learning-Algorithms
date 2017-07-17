import numpy as np
from scipy import linalg as LA


class Mds(object):
    # D:distance matrix, item ojn i, represents distance of xi and xj

    def __init__(self, D, d_):
        self.d_ = d_
        self.D = D
        self.m = D.shape[0]
        self.B = np.zeros((self.m, self.m))
        self.new_sample_mat = np.zeros((self.d_, self.d_))

    def calcu_B(self):

        def dist_ij2(i, j):
            return self.D[i][j]**2

        def dist_i2(i):
            sum = 0.0
            for j in range(self.m):
                sum += self.D[i][j]**2
            result = (1 / self.m) * sum
            return result

        def dist_j2(j):
            sum = 0.0
            for i in range(self.m):
                sum += self.D[i][j]**2
            result = (1 / self.m) * sum
            return result

        def dist__2():
            sum = 0.0
            for i in range(self.m):
                for j in range(self.m):
                    sum += self.D[i][j]**2
                result = (1 / self.m**2) * sum
            return result

        for i in range(self.m):
            for j in range(self.m):
                d_ij2 = dist_ij2(i, j)
                d_i2 = dist_i2(i)
                d_j2 = dist_j2(j)
                d__2 = dist__2()
                self.B[i][j] = -0.5 * (d_ij2- d_i2 - d_j2 + d__2)
        return self.B

    def eigen_decomp(self, d_):
        e_vals, e_vecs = LA.eig(self.B)
        diag_mat_ori = np.diag(tuple(e_vals))
        diag_mat_d_ = diag_mat_ori[: int(d_), :int(d_)]
        eigen_vec_mat = e_vecs[:, :d_]
        self.new_sample_mat = eigen_vec_mat * np.sqrt(diag_mat_d_)
        return self.new_sample_mat
