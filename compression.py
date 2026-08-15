from itertools import product
import numpy as np
from scipy.fft import dctn, idctn
from skimage.util.shape import view_as_blocks
from huffman import canonical_huffman, encode, pack_bits
from hilbertcurve.hilbertcurve import HilbertCurve
import struct

class Compression:
  def __init__(self, arr: np.ndarray, dimension: int) -> None:
    self.arr = arr
    self.dimension = dimension
    self._block_size = 8
    
  def encode(self, k: int):
    blocks = self.split_into_blocks()
    self._data_size = blocks.shape[:-self.dimension]
    self._indices = list(product(*[range(size) for size in self._data_size]))

    dct_blocks = self.dct_for_each_box(blocks)
    quantized_blocks = dct_blocks
    for index in self._indices:
      quantized_blocks[*index] = self.quantize(quantized_blocks[*index], k)

    delta_dc = self.difference_for_dc(quantized_blocks)

    points = self.hilbert_curve_for_a_box()

    run_length = []
    for index in self._indices:
      block = quantized_blocks[*index]
      elements = np.array([block[*point] for point in points[1:]])
      run, length = self.rle(elements)
      run_length += list(zip([int(x) for x in run],[int(x) for x in length]))

    dc_coded, dc_size, dc_code_table = self.huffman(delta_dc)
    ac_coded, ac_size, ac_code_table = self.huffman(run_length)

    pairs_to_bytes = lambda p: b''.join(a.to_bytes(4, 'big', signed=True) + b.to_bytes(4, 'big', signed=True) for a, b in p)

    final_coded = dc_size.to_bytes(2, 'big') + dc_coded \
      + ac_size.to_bytes(4, 'big') + ac_coded \
      + struct.pack(f'>{len(dc_code_table[0])}h', *dc_code_table[0]) + struct.pack(f'>{len(dc_code_table[1])}h', *dc_code_table[1]) \
      + struct.pack(f'>{len(ac_code_table[0])}h', *ac_code_table[0]) + pairs_to_bytes(ac_code_table[1]) \
      + k.to_bytes(1, 'big') # type: ignore

    return final_coded


  def split_into_blocks(self) -> np.ndarray:
    return view_as_blocks(self.arr, (self._block_size,) * self.dimension)

  def dct_for_each_box(self, blocks) -> np.ndarray:
    return dctn(blocks, norm='ortho') # type: ignore

  def quantize(self, block: np.ndarray, k: int) -> np.ndarray:
    idx = [slice(None)] * self.dimension
    idx = tuple(slice(0, min(k, s)) for s in block.shape)
    
    b = np.zeros_like(block)
    b[idx] = block[idx]
    return b.astype(int)

  def difference_for_dc(self, blocks) -> list:

    prev = 0 # previous element
    ds = [] # final result (delta_dc)

    for index in self._indices:
      block = blocks[*index] # get a block
      dc = int(block[*((0,)*self.dimension)]) # get (0, ..., 0) element from the block
      diff = dc - prev # difference with previous element
      prev = dc # update previous element
      ds.append(diff)

    return ds

  def huffman(self, data: list):
    bits, huffval = canonical_huffman(data)
    encoded = encode(data, bits, huffval)
    packed, n = pack_bits(encoded)
    return packed, n, (bits, huffval)

  def hilbert_curve_for_a_box(self):
    p = 3 # log2(self.block_size)
    hilbert_curve = HilbertCurve(p, self.dimension)
    distances = list(range(2 ** (p * self.dimension)))
    return list(hilbert_curve.points_from_distances(distances))

  def rle(self, sequence):
    comp_seq_index, = np.concatenate(([True], sequence[1:] != sequence[:-1], [True])).nonzero()
    return sequence[comp_seq_index[:-1]], np.ediff1d(comp_seq_index)
