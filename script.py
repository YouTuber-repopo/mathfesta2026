from itertools import product
import numpy as np
from scipy.fft import dctn, idctn
from skimage.util.shape import view_as_blocks
from dahuffman import HuffmanCodec
from hilbertcurve.hilbertcurve import HilbertCurve

x_size = 16
y_size = 24
z_size = 16
w_size = 32

block_size = 8

data1 = np.random.rand(w_size, z_size, y_size, x_size)
data2 = np.arange(x_size * y_size).reshape(y_size, x_size)
data3 = np.arange(x_size * y_size * z_size).reshape(z_size, y_size, x_size)

# print(data1.shape)
# print(data2.shape)
# print(data3.shape)
# print(data3)
# print(data2)


# blocks = view_as_blocks(data2, block_shape=(block_size, block_size))
# print(data2.shape)
# print(blocks.shape)
# print(data2)
# print(blocks)


class Compression:
  def __init__(self, arr: np.ndarray, dimension: int) -> None:
    self.arr = arr
    self.dimension = dimension
    self.block_size = 8
    self.blocks = self.split_into_blocks()
    self.data_size = self.blocks.shape[:-self.dimension]
    self.t = self.dct_for_each_box()
    print(self.blocks.shape)
    print(self.t.shape)
    # print(self.t)
    # print(np.allclose(self.blocks, idctn(self.t, norm='ortho')))
    
  def split_into_blocks(self) -> np.ndarray:
    return view_as_blocks(self.arr, (self.block_size,) * self.dimension)

  def dct_for_each_box(self) -> np.ndarray:
    return dctn(self.blocks, norm='ortho')

  def quantize(self, block: np.ndarray) -> np.ndarray:
    return block

  def difference_for_dc(self) -> list:
    indices = product(*[range(size) for size in self.data_size])

    diff = 0
    ds = []

    for index in indices:
      block = self.blocks[*index]
      dc = int(block[*((0,)*self.dimension)])
      diff = dc - diff
      ds.append(diff)

    return ds

  def huffman_dc(self, delta_dc: list):
    self.dc_codec = HuffmanCodec.from_data(delta_dc)
    return self.dc_codec.encode(delta_dc)



c2 = Compression(data2, 2)
c2.difference_for_dc()
# c3 = Compression(data3, 3)
# c3.dct_for_each_box()

print('-------')
# print(c3.blocks[0,0,0])