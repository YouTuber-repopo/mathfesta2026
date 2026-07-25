from itertools import product
import numpy as np
from scipy.fft import dctn, idctn
from skimage.util.shape import view_as_blocks
from dahuffman import HuffmanCodec
from hilbertcurve.hilbertcurve import HilbertCurve

class Compression:
  def __init__(self, arr: np.ndarray, dimension: int) -> None:
    self.arr = arr
    self.dimension = dimension
    self.block_size = 8
    self.blocks = self.split_into_blocks()
    self.data_size = self.blocks.shape[:-self.dimension]
    self.dct_blocks = self.dct_for_each_box()
    
  def split_into_blocks(self) -> np.ndarray:
    return view_as_blocks(self.arr, (self.block_size,) * self.dimension)

  def dct_for_each_box(self) -> np.ndarray:
    return dctn(self.blocks, norm='ortho')

  def quantize(self, block: np.ndarray) -> np.ndarray:
    return block

  def difference_for_dc(self) -> list:
    indices = product(*[range(size) for size in self.data_size])

    prev = 0
    ds = []

    for index in indices:
      block = self.dct_blocks[*index]
      dc = int(block[*((0,)*self.dimension)])
      diff = dc - prev
      prev = dc
      ds.append(diff) 

    return ds

  def huffman_dc(self, delta_dc: list):
    self.dc_codec = HuffmanCodec.from_data(delta_dc)
    return self.dc_codec.encode(delta_dc)

  def hilbert_curve_for_a_box(self) -> np.ndarray:
    p = 3 # log2(self.block_size)
    hilbert_curve = HilbertCurve(p, self.dimension)
    distances = list(range(2 ** (p * self.dimension)))
    return hilbert_curve.points_from_distances(distances)