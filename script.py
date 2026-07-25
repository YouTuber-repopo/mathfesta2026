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

  # print('--------------------------------')
  # print(c.blocks)
  # print('--------------------------------')
  # print(c.dct_blocks)
  print('--------------------------------')
  print(c.dct_blocks.shape)
  print('--------------------------------')
  delta_dc = c.difference_for_dc()
  print(delta_dc)
  print('--------------------------------')
  encoded = c.huffman_dc(delta_dc)
  print(list(encoded))
  print(c.dc_codec.print_code_table())
  print(c.dc_codec.decode(encoded))
  print('--------------------------------')
  print(c.hilbert_curve_for_a_box())
  print('--------------------------------')