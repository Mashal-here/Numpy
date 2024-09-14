# Numpy
a) Create a 1D array of numbers from 1 to 10.
import numpy as np
array = np.arange(1, 11)
print(array)
b) Reshape the array into a 2x5 matrix.
matrix = array.reshape(2, 5)
print(matrix)
c) Find the element located in the second row, third column of the matrix.
element = matrix[1, 2]
print(element)
2. Basic Array Operations
a) Create two NumPy arrays:
array_1 = np.array([10, 20, 30])
array_2 = np.array([1, 2, 3])
b) Perform element-wise addition, subtraction, and multiplication.
addition = array_1 + array_2
subtraction = array_1 - array_2
multiplication = array_1 * array_2

print(addition)
print(subtraction)
print(multiplication)
c) Calculate the sum, mean, and standard deviation of the resulting array after element-wise multiplication.
sum_result = np.sum(multiplication)
mean_result = np.mean(multiplication)
std_dev = np.std(multiplication)

print(sum_result)
print(mean_result)
print(std_dev)
3. Broadcasting and Reshaping
a) Create a 1D array of numbers from 0 to 15 and reshape it into a 3x5 matrix.
array = np.arange(0, 15).reshape(3, 5)
print(array)
b) Add 10 to each element in the matrix using broadcasting.
broadcasted_matrix = array + 10
print(broadcasted_matrix)
c) Create another 3x5 matrix and multiply the two matrices using broadcasting.
matrix_2 = np.ones((3, 5)) * 2  # Another 3x5 matrix
result_matrix = broadcasted_matrix * matrix_2
print(result_matrix)

Here are detailed answers to each question related to NumPy:

Section 1: Basic Questions
1. NumPy Array Creation and Manipulation
a) Create a 1D array of numbers from 1 to 10.

import numpy as np
array = np.arange(1, 11)
print(array)
b) Reshape the array into a 2x5 matrix.

matrix = array.reshape(2, 5)
print(matrix)
c) Find the element located in the second row, third column of the matrix.


element = matrix[1, 2]
print(element)
2. Basic Array Operations
a) Create two NumPy arrays:


array_1 = np.array([10, 20, 30])
array_2 = np.array([1, 2, 3])
b) Perform element-wise addition, subtraction, and multiplication.


addition = array_1 + array_2
subtraction = array_1 - array_2
multiplication = array_1 * array_2

print(addition)
print(subtraction)
print(multiplication)
c) Calculate the sum, mean, and standard deviation of the resulting array after element-wise multiplication.


sum_result = np.sum(multiplication)
mean_result = np.mean(multiplication)
std_dev = np.std(multiplication)

print(sum_result)
print(mean_result)
print(std_dev)
3. Broadcasting and Reshaping
a) Create a 1D array of numbers from 0 to 15 and reshape it into a 3x5 matrix.


array = np.arange(0, 15).reshape(3, 5)
print(array)
b) Add 10 to each element in the matrix using broadcasting.


broadcasted_matrix = array + 10
print(broadcasted_matrix)
c) Create another 3x5 matrix and multiply the two matrices using broadcasting.
matrix_2 = np.ones((3, 5)) * 2  # Another 3x5 matrix
result_matrix = broadcasted_matrix * matrix_2
print(result_matrix)
4. Logical Operations and Boolean Indexing
array = np.array([15, 22, 33, 41, 50, 65, 72])
b) Use boolean indexing to filter out values that are greater than 40.
filtered_values = array[array > 40]
print(filtered_values)
c) Replace all values greater than 40 with the value 0 in the original array.
array[array > 40] = 0

1. Matrix Operations
a) Create a 3x3 matrix using the values [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(array)
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix)
b) Find the transpose of the matrix
transpose = np.transpose(matrix)
print(transpose)
c) Perform matrix multiplication with another 3x3 matrix using np.dot().
matrix_2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
result_matrix = np.dot(matrix, matrix_2)
print(result_matrix)
d) Calculate the inverse of the resulting matrix, if possible.
if np.linalg.det(result_matrix) != 0:  # Check if matrix is invertible
    inverse_matrix = np.linalg.inv(result_matrix)
    print(inverse_matrix)
else:
    print("Matrix is not invertible")
   2. Statistical Analysis
      array_x = np.array([1, 3, 5, 7, 9])
array_y = np.array([2, 4, 6, 8, 10])

Here are detailed answers to each question related to NumPy:

Section 1: Basic Questions
1. NumPy Array Creation and Manipulation
a) Create a 1D array of numbers from 1 to 10.

python
Copy code
import numpy as np
array = np.arange(1, 11)
print(array)
b) Reshape the array into a 2x5 matrix.

python
Copy code
matrix = array.reshape(2, 5)
print(matrix)
c) Find the element located in the second row, third column of the matrix.

python
Copy code
element = matrix[1, 2]
print(element)
2. Basic Array Operations
a) Create two NumPy arrays:

python
Copy code
array_1 = np.array([10, 20, 30])
array_2 = np.array([1, 2, 3])
b) Perform element-wise addition, subtraction, and multiplication.

python
Copy code
addition = array_1 + array_2
subtraction = array_1 - array_2
multiplication = array_1 * array_2

print(addition)
print(subtraction)
print(multiplication)
c) Calculate the sum, mean, and standard deviation of the resulting array after element-wise multiplication.

python
Copy code
sum_result = np.sum(multiplication)
mean_result = np.mean(multiplication)
std_dev = np.std(multiplication)

print(sum_result)
print(mean_result)
print(std_dev)
3. Broadcasting and Reshaping
a) Create a 1D array of numbers from 0 to 15 and reshape it into a 3x5 matrix.

python
Copy code
array = np.arange(0, 15).reshape(3, 5)
print(array)
b) Add 10 to each element in the matrix using broadcasting.

python
Copy code
broadcasted_matrix = array + 10
print(broadcasted_matrix)
c) Create another 3x5 matrix and multiply the two matrices using broadcasting.

python
Copy code
matrix_2 = np.ones((3, 5)) * 2  # Another 3x5 matrix
result_matrix = broadcasted_matrix * matrix_2
print(result_matrix)
4. Logical Operations and Boolean Indexing
a) Create an array with values: [15, 22, 33, 41, 50, 65, 72].

python
Copy code
array = np.array([15, 22, 33, 41, 50, 65, 72])
b) Use boolean indexing to filter out values that are greater than 40.

python
Copy code
filtered_values = array[array > 40]
print(filtered_values)
c) Replace all values greater than 40 with the value 0 in the original array.

python
Copy code
array[array > 40] = 0
print(array)
Section 2: Advanced Questions
1. Matrix Operations
a) Create a 3x3 matrix using the values [[1, 2, 3], [4, 5, 6], [7, 8, 9]].

python
Copy code
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix)
b) Find the transpose of the matrix.

python
Copy code
transpose = np.transpose(matrix)
print(transpose)
c) Perform matrix multiplication with another 3x3 matrix using np.dot().

python
Copy code
matrix_2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
result_matrix = np.dot(matrix, matrix_2)
print(result_matrix)
d) Calculate the inverse of the resulting matrix, if possible.

python
Copy code
if np.linalg.det(result_matrix) != 0:  # Check if matrix is invertible
    inverse_matrix = np.linalg.inv(result_matrix)
    print(inverse_matrix)
else:
    print("Matrix is not invertible")
2. Statistical Analysis
a) Create two arrays: array_x = [1, 3, 5, 7, 9] and array_y = [2, 4, 6, 8, 10].

python
Copy code
array_x = np.array([1, 3, 5, 7, 9])
array_y = np.array([2, 4, 6, 8, 10])
b) Compute the correlation coefficient between the two arrays using np.corrcoef().
correlation = np.corrcoef(array_x, array_y)
print(correlation)

c) Generate 1000 random numbers from a normal distribution and find the 95th percentile.
random_numbers = np.random.normal(0, 1, 1000)
percentile_95 = np.percentile(random_numbers, 95)
print(percentile_95)
3. Solving Linear Equations
coefficients = np.array([[2, 1], [1, -1]])
constants = np.array([10, 2])
solution = np.linalg.solve(coefficients, constants)
print(solution)
4. Fancy Indexing and Conditional Selection
array = np.array([25, 45, 15, 75, 35, 55, 85])
b) Use fancy indexing to rearrange the elements in descending order.
sorted_array = array[np.argsort(array)[::-1]]
print(sorted_array)
c) Use np.where() to replace all values greater than 50 with 100 and values less than or equal to 50 with 0.
modified_array = np.where(array > 50, 100, 0)
print(modified_array)
5. Performance Comparison: NumPy vs. Python Lists
a) Create a Python list with 1 million elements and a NumPy array with the same number of elements.
python_list = list(range(1000000))
numpy_array = np.arange(1000000)
b) Write a Python function that computes the sum of all elements in both the list and the NumPy array.
def sum_elements(data):
    return sum(data)

list_sum = sum_elements(python_list)
numpy_sum = np.sum(numpy_array)

print(list_sum)
print(numpy_sum)
c) Use the timeit module to compare the time taken to sum the elements in both the list and the NumPy array. Report the time difference.
import timeit

list_time = timeit.timeit(lambda: sum_elements(python_list), number=10)
numpy_time = timeit.timeit(lambda: np.sum(numpy_array), number=10)

print(f"Python list time: {list_time}")
print(f"NumPy array time: {numpy_time}")





