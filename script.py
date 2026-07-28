from compression import Compression
import numpy as np

x_size = 16
y_size = 24
z_size = 16
w_size = 32

block_size = 8

data1 = np.random.rand(w_size, z_size, y_size, x_size)
data2 = np.arange(x_size * y_size).reshape(y_size, x_size)
data3 = np.arange(x_size * y_size * z_size).reshape(z_size, y_size, x_size)


if __name__ == '__main__':
  c = Compression(data2, 2)

  np.set_printoptions(precision=2, linewidth=100)

  compressed = c.encode(7)
  print(len(str(data2)))
  print(len(compressed))