from compression import Compression
import numpy as np
import sys

def find_compression_rate(data_size, dim):
  shape = (data_size,) * dim
  rand = np.random.random_sample(shape)
  norm = np.random.randn(*shape)

  cr = Compression(rand, dim)
  cn = Compression(norm, dim)

  random_reduction_rates = []
  normal_reduction_rates = []

  float_byte = 4
  int_byte = 4

  random_size = rand.nbytes
  normal_size = norm.nbytes

  for k in range(6, 9):
    r_compressed = cr.encode(k)
    n_compressed = cn.encode(k)
    random_reduction_rates.append((random_size - r_compressed * int_byte) / (random_size))
    normal_reduction_rates.append((normal_size - n_compressed * int_byte)  / (normal_size))

  return random_reduction_rates, normal_reduction_rates

if __name__ == '__main__':
  seed = int(input("Enter a seed: "))
  np.random.seed(seed)
  for d in range(1,6):
    for data_size in [8, 16, 24, 32]:
      random_reduction_rates, normal_reduction_rates = find_compression_rate(data_size, d)
      print(f"Dimension: {d}")
      print(f"Data size: {data_size}")
      print(f"Random reduction rates: {random_reduction_rates}")
      print(f"Normal reduction rates: {normal_reduction_rates}")
      print("--------------------------------")