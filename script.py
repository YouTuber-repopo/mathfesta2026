from compression import Compression
import numpy as np
import sys
import csv

def find_compression_rate(data_size, dim):
  shape = (data_size,) * dim

  random_reduction_rates = []
  normal_reduction_rates = []

  for seed in range(10):
    np.random.seed(seed)

    rand = np.random.random_sample(shape)
    norm = np.random.randn(*shape)

    cr = Compression(rand, dim)
    cn = Compression(norm, dim)

    random_size = rand.nbytes
    normal_size = norm.nbytes

    for k in range(6, 9):
      r_compressed = cr.encode(k)
      n_compressed = cn.encode(k)
      random_reduction_rate = (random_size - len(r_compressed)) / (random_size)
      normal_reduction_rate = (normal_size - len(n_compressed))  / (normal_size)
      random_reduction_rates.append(random_reduction_rate)
      normal_reduction_rates.append(normal_reduction_rate)

  return random_reduction_rates, normal_reduction_rates

if __name__ == '__main__':
  random_file = open('random_output.csv', 'w', newline='')
  normal_file = open('normal_output.csv', 'w', newline='')
  random_writer = csv.writer(random_file)
  normal_writer = csv.writer(normal_file)

  for d in range(1,6):
    print()
    print('------')
    print(f'Dim {d}:', end=' \n')
    for data_size in [8, 16, 24, 32]:
      random_reduction_rates, normal_reduction_rates = find_compression_rate(data_size, d)

      for i in range(10):
        random_writer.writerow([d, data_size] + random_reduction_rates[i*3:i*3+3])
        normal_writer.writerow([d, data_size] + normal_reduction_rates[i*3:i*3+3])
      print(data_size)


  random_file.close()

  # dim = 6
  # data = np.random.random_sample((32, ) * dim)
  #
  # print(f'Data shape: {data.shape}')
  # print(f'Data size: {data.nbytes} bytes')
  #
  # compresser = Compression(data, dim)
  # k = 6
  #
  # print('--------------------------------')
  # print(f'Encoding with k={k}')
  #
  # encoded = compresser.encode(k)
  #
  # print('--------------------------------')
  #
  # print(f'Encoded: {encoded}')
  # print(f'Encoded list: {list(encoded)}')
  # print(f'Encoded size: {len(encoded)} bytes')
