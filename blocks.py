# Block Partition class and test function
# Written by Nicholas Hausch January 2020


import math
import sys
import copy
import time
import numpy as np
from itertools import combinations


# Euclidean distance in arbitrary dimensions
def Euclidean(p1, p2):
    total = 0
    for i in range(p1.size):
        total = total + math.pow(p1[i] - p2[i], 2)
    return math.sqrt(total)


# Get the max value in a dictionary
def MaxValue(dict):
    v = list(dict.values())
    k = list(dict.keys())
    max_v = max(v)
    return k[v.index(max_v)], max_v


# Computes the distance vector for the given data point
def ComputeDistanceVector(data, iv_index):
  num_points = np.size(data, 0)
  distance_vector = np.zeros((num_points))
  for j in range(0, num_points):
    distance = Euclidean(data[iv_index], data[j])
    distance_vector[j] = distance
  return distance_vector

# Computes the distance matrix
def ComputeDistanceMatrix(data):
  num_points = np.size(data, 0)
  distance_matrix = np.zeros((num_points, num_points))
  for i in range(num_points):
      for j in range(i + 1, num_points):
          distance = Euclidean(data[i], data[j])
          distance_matrix[i][j] = distance
  return distance_matrix + np.transpose(distance_matrix)


# Block Partition class
class BlockPartition:

  # Constructor
  def __init__(self, data, partition_size):
    self.data = data
    self.dimensions = data.shape[1]
    self.partition_size = partition_size
    self.lower_bounds = np.amin(self.data, axis=0)
    self.upper_bounds = np.amax(self.data, axis=0)
    self.cache = np.zeros((np.size(data, axis=0), (np.size(data, axis=0))))

    # Get deltas
    self.deltas = np.zeros((self.lower_bounds.size))
    for i in range(self.deltas.size):
      self.deltas[i] = (self.upper_bounds[i] - self.lower_bounds[i]) / partition_size

    # Get intervals
    self.intervals = {}
    for i in range(self.deltas.size):
      self.intervals[i] = np.zeros((self.partition_size))
      delta = self.deltas[i]
      lower_bound = self.lower_bounds[i]
      for j in range(self.partition_size):
        self.intervals[i][j] = lower_bound + delta * j

    # Assign each point to a block
    self.assignments = {}
    self.blocks = {}
    for i in range(np.size(data, 0)):
      point = data[i, :]
      self.assignments[i] = np.zeros(len(self.intervals))
      for j in range(point.size):
          index = self.BinarySearch(point[j], j, 0, len(self.intervals[j]))
          self.assignments[i][j] = index
      assignment_tuple = tuple(self.assignments[i])
      if (assignment_tuple in self.blocks):
          self.blocks[assignment_tuple].append(i)
      else:
          self.blocks[assignment_tuple] = [i]

    # Get index combinations
    self.indexes = np.arange(self.dimensions)
    self.index_combinations = {}
    self.index_combination_compliments = {}
    self.negative_subindex_combinations = {}
    for sublevel in range(self.dimensions):

      # Get indexes
      self.index_combinations[sublevel] = list(combinations(self.indexes, sublevel))
      
      # Get complimentary set of indexes
      self.index_combination_compliments[sublevel] = []
      for index_comb in self.index_combinations[sublevel]:
        complimentary_indexes = []
        for i in range(self.dimensions):
          if i not in index_comb:
            complimentary_indexes.append(i)
        self.index_combination_compliments[sublevel].append(complimentary_indexes)

      # Get negative index combinations for 
      self.negative_subindex_combinations[sublevel] = []
      negative_indexes = np.arange(sublevel)
      for num_negative in range(1, sublevel + 1):
        self.negative_subindex_combinations[sublevel].extend(list(combinations(negative_indexes, num_negative)))


  # Recursive ranged binary search
  def BinarySearch(self, value, dim, left, right):

    # Sanity check
    if (right == left):
        return right 

    # Get index
    index = math.floor((right + left) / 2)
    if (self.intervals[dim][index] < value and (self.intervals[dim][index] + self.deltas[dim]) > value):
        return index
    elif (self.intervals[dim][index] == value):
            return index
    elif ((self.intervals[dim][index] + self.deltas[dim]) == value):
            return index
    elif (self.intervals[dim][index] < value):
        return self.BinarySearch(value, dim, index + 1, right)
    else:
        return self.BinarySearch(value, dim, left, index - 1)


  # Compute the nearest neighbors of the data point with the given index
  def NearestNeighbors(self, iv_index, N):
    
    # Get the block associated with the point
    root_block = tuple(self.assignments[iv_index])

    # Compute the distances to other points in the root block
    distances = {}
    new_neighbor = False
    max_index = -1
    max_dist = 0.0
    for comp_index in self.blocks[root_block]:
      if (comp_index != iv_index):
        distance = self.GetDistance(iv_index, comp_index)
        new_neighbor, max_index, max_dist = self.RegisterDistance(distances, distance, comp_index, max_index, max_dist, N)

    # Calculate the distance to the other blocks
    min_distances = []
    max_distances = []
    blocks_buffer = []
    for block in self.blocks:
      min_distance, max_distance = self.BlockDistance(root_block, block)

      # Compare the points if distance could be 0
      if (min_distance == 0):
        for comp_index in self.blocks[block]:
          if (comp_index != iv_index):
            distance = self.GetDistance(iv_index, comp_index)
            new_neighbor, max_index, max_dist = self.RegisterDistance(distances, distance, comp_index, max_index, max_dist, N)
      else:
        min_distances.append(min_distance)
        max_distances.append(max_distance)
        blocks_buffer.append(block)

    # Sort by the max distance 
    max_arr = np.array(min_distances)
    sorted_indexes = np.argsort(max_arr, axis=None)

    # Compute the distances to the other blocks
    for index in sorted_indexes:
      if (min_distances[index] > max_dist and len(distances) == N):
        break
      for comp_index in self.blocks[blocks_buffer[index]]:
        if (comp_index != iv_index):
          distance = self.GetDistance(iv_index, comp_index)
          new_neighbor, max_index, max_dist = self.RegisterDistance(distances, distance, comp_index, max_index, max_dist, N)


  # Computes the min and max distance between two blocks
  def BlockDistance(self, block, comp_block):
    min_sum = 0
    max_sum = 0
    for i in range(self.dimensions):
      lower_bound, upper_bound = self.GetBounds(block[i], comp_block[i], self.deltas[i])
      min_sum = min_sum + math.pow(lower_bound, 2)
      max_sum = max_sum + math.pow(upper_bound, 2)
    return math.sqrt(min_sum), math.sqrt(max_sum)


  # Computes the min and max distance between two indexes given a delta
  def GetBounds(self, index, comp_index, delta):
    if (index == comp_index):
      return 0, delta
    elif (index > comp_index):
      return (index - (comp_index + 1)) * delta, ((index + 1) - comp_index) * delta
    elif (index < comp_index):
      return (comp_index - (index + 1)) * delta, ((comp_index + 1) - index) * delta


  # Adds a distance to the map of nearest neighbors and their distances, and tracks the max index and value
  def RegisterDistance(self, distances, distance, comp_index, max_index, max_dist, N):

    # We do not have the N neighbors yet
    if (len(distances) < N):
      distances[comp_index] = distance
      max_index, max_dist = MaxValue(distances)
      return True, max_index, max_dist

    # We have N neighbors but comp_index is closer than the max
    elif (distance < max_dist):
      del distances[max_index]
      distances[comp_index] = distance
      max_index, max_dist = MaxValue(distances)
      return True, max_index, max_dist

    # We already have N better neighbors
    return False, max_index, max_dist


  # Get distance between two points (compute or from cache)
  def GetDistance(self, index1, index2):
    if (self.cache[index1][index2] != 0):
      return self.cache[index1][index2]
    else:
      distance = Euclidean(self.data[index1], self.data[index2])
      self.cache[index1][index2] = distance
      self.cache[index2][index1] = distance
      return distance


# Main test function
def Main():
  dim = 3
  lower_bound = 0
  upper_bound = 100
  num_samples = 20000
  partition_size = 6
  test_index = 1
  second_test_index = 500
  N = 15
  data = np.random.rand(num_samples, dim) * upper_bound - lower_bound

  # Get partition
  partition_start = time.time()
  bp = BlockPartition(data, partition_size)
  partition_end = time.time()
  print("** Partition:", (partition_end - partition_start) * 1000, "nsecs.")

  # Get nearest neighbors 
  nn_start = time.time()
  bp.NearestNeighbors(test_index, N)
  nn_end = time.time()
  print("** Single Run:", (nn_end - nn_start) * 1000, "nsecs.")

  nn_start = time.time()
  bp.NearestNeighbors(second_test_index, N)
  nn_end = time.time()
  print("** Single Run:", (nn_end - nn_start) * 1000, "nsecs.")

  # Verify the results
  distance_start = time.time()
  distance_vector = ComputeDistanceVector(data, test_index)
  sorted_vector = np.argsort(distance_vector)
  for i in range(1, N + 1):
    print(sorted_vector[i], distance_vector[sorted_vector[i]])
  distance_end = time.time()
  print("** Brute Force:", (distance_end - distance_start) * 1000, "nsecs.")

  distance_start = time.time()
  distance_matrix = ComputeDistanceMatrix(data)
  distance_end = time.time()
  print("** Distance Matrix:", (distance_end - distance_start) * 1000, "nsecs.")

  

if __name__ == '__main__':
    Main()




