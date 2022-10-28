import numpy as np

#NumPy Array
#How to create an empty and a full NumPy array?
emptyArray = np.empty((3, 4))
fullArray = np.full((3, 4), 7)

print(emptyArray)
print(fullArray)

#Create a Numpy array filled with all zeros
allZeros = np.zeros((3, 4))

#Create a Numpy array filled with all ones
allOnes = np.ones((3, 4))

#Check whether a Numpy array contains a specified row
num = np.arange(20)
mainArray = np.reshape(num, [4, 5])

test1 = [0, 1, 2, 3, 4]
test2 = [5, 6, 7, 8, 9]
test3 = [10, 11, 12, 13, 15]
test4 = [16, 17, 18, 19, 19]

print(test1 in mainArray)
print(test2 in mainArray)
print(test3 in mainArray)
print(test4 in mainArray)

#How to Remove rows in Numpy array that contains non-numeric values?
mainArray = np.array([[1, 2, 3], [4, 5, np.nan], [7, 8, 9], [10, False, True]])
print(mainArray[~np.isnan(mainArray).any(axis=1)])

#Remove single-dimensional entries from the shape of an array
mainArray = np.zeros((5, 1, 4))
print(np.squeeze(mainArray).shape)

#Find the number of occurrences of a sequence in a NumPy array
mainArray = np.array([[1, 2, 3, 4], 
                      [2, 3, 2, 3],
                      [1, 3, 2, 4],
                      [4, 3, 2, 3]])

print(repr(mainArray).count("2, 3"))
  
#Find the most frequent value in a NumPy array
mainArray = np.array([1, 2, 3, 1, 2, 1, 1, 1, 3, 2, 2, 1])
print(np.bincount(mainArray).argmax())

#Combining a one and a two-dimensional NumPy Array
array1D = np.arange(4)
array2D = np.arange(8).reshape(2, 4)

for i, j in np.nditer([array1D, array2D]):
    print("%d:%d" % (i, j),)
    
#How to build an array of all combinations of two NumPy arrays?
array1 = [1, 2, 3]
array2 = [4, 5]

print(np.array(np.meshgrid(array1, array2)).T.reshape(-1,3))

#How to add a border around a NumPy array?
mainArray = np.ones((3, 3))

print(np.pad(mainArray, pad_width=1, mode='constant', constant_values=0))

#How to compare two NumPy arrays?
array1 = np.array([[1, 2], [3, 4]])
array2 = np.array([[1, 2], [3, 4]])

print((array1 == array2).all())
  
#How to check whether specified values are present in NumPy array?
mainArray = np.array([[1.12, 2.0, 3.45], [2.33, 5.12, 6.0]], float) 

print(0 in mainArray)
print(6 in mainArray)
print(2.3 in mainArray)

#How to get all 2D diagonals of a 3D NumPy array?
mainArray = np.arange(3*4*5).reshape(3,4,5)

result = np.diagonal(mainArray, axis1=1, axis2=2)
print("\n2D diagonals: ", result)

#Flatten a Matrix in Python using NumPy
# declare matrix with np
mainMatrix = np.array([[2, 3], [4, 5]])
  
print(mainMatrix.flatten())

#Flatten a 2d numpy array into 1d array
mainArray = np.array([[1, 2, 3], 
                      [4, 5, 6], 
                      [7, 8, 9]])
  
print(mainArray.flatten())

#Move axes of an array to new positions
mainArray = np.zeros((2, 3, 4))

print(np.moveaxis(mainArray, 0, -1).shape)
print(np.moveaxis(mainArray, -1, 0).shape)

#Interchange two axes of an array
mainArray = np.array([[1, 2, 3]])

print(np.swapaxes(mainArray, 0, 1))

#NumPy – Fibonacci Series using Binet Formula

#Counts the number of non-zero values in the array
#Count the number of elements along a given axis
#Trim the leading and/or trailing zeros from a 1-D array
#Change data type of given numpy array
#Reverse a numpy array
mainArray = np.array([1, 2, 3, 6, 4, 5])
   
print(mainArray[::-1])
  
#How to make a NumPy array read-only?
mainArray = np.zeros(10)
mainArray.flags.writeable = False
mainArray[0] = 1


#Questions on NumPy Matrix
mainMatrix = np.matrix([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                        [10, 20, 30]])

secondMatrix = np.matrix([[10, 20, 30],
                          [9, 8, 7],
                          [6, 5, 4],
                          [3, 2, 1]])

#Get the maximum value from given matrix
mainMatrix.max()

#Get the minimum value from given matrix
mainMatrix.min()

#Find the number of rows and columns of a given matrix using NumPy
mainMatrix.shape()

#Select the elements from a given matrix
mainMatrix[1:-1,2]

#Find the sum of values in a matrix
mainMatrix.sum()

#Calculate the sum of the diagonal elements of a NumPy array
np.diagonal(mainMatrix, axis1=0, axis2=1).sum()

#Adding and Subtracting Matrices in Python
print(np.add(mainMatrix, secondMatrix))
print(np.subtract(mainMatrix, secondMatrix))

#Ways to add row/columns in numpy array
addColumns = np.array([[1], [2], [3], [4]])
 
print(np.append(mainArray, addColumns, axis=1))
 
#Matrix Multiplication in NumPy
print(np.matmul(mainMatrix, secondMatrix))
print(np.dot(mainMatrix, secondMatrix))

#Get the eigen values of a matrix
print(np.linalg.eig(mainMatrix)[0])

#How to Calculate the determinant of a matrix using NumPy?
print(np.linalg.det(mainMatrix))

#How to inverse a matrix using NumPy
print(np.linalg.inv(mainMatrix))

#How to count the frequency of unique values in NumPy array?
mainArray = np.array([10, 10, 20, 10, 20, 20, 20, 30, 30, 50, 40, 40])

uniqueElements, countsElements = np.unique(mainArray, return_counts=True)
print(np.asarray((uniqueElements, countsElements)))

#Multiply matrices of complex numbers using NumPy in Python

#Compute the outer product of two given vectors using NumPy in Python
#Calculate inner, outer, and cross products of matrices and vectors using NumPy
#Compute the covariance matrix of two given NumPy arrays
#Convert covariance matrix to correlation matrix using Python
#Compute the Kronecker product of two mulitdimension NumPy arrays
#Convert the matrix into a list




#Questions on NumPy Indexing
#Replace NumPy array elements that doesn’t satisfy the given condition
#Return the indices of elements where the given condition is satisfied
#Replace NaN values with average of columns
#Replace negative value with zero in numpy array
#How to get values of an NumPy array at certain index positions?
#Find indices of elements equal to zero in a NumPy array
#How to Remove columns in Numpy array that contains non-numeric values?
#How to access different rows of a multidimensional NumPy array?
#Get row numbers of NumPy array having element larger than X
#Get filled the diagonals of NumPy array
#Check elements present in the NumPy array
#Combined array index by index



#Questions on NumPy Linear Algebra
#Find a matrix or vector norm using NumPy
#Calculate the QR decomposition of a given matrix using NumPy
#Compute the condition number of a given matrix using NumPy
#Compute the eigenvalues and right eigenvectors of a given square array using NumPy?
#Calculate the Euclidean distance using NumPy
#Questions on NumPy Random
#Create a Numpy array with random values
#How to choose elements from the list with different probability using NumPy?
#How to get weighted random choice in Python?
#Generate Random Numbers From The Uniform Distribution using NumPy
#Get Random Elements form geometric distribution
#Get Random elements from Laplace distribution
#Return a Matrix of random values from a uniform distribution
#Return a Matrix of random values from a Gaussian distribution



#Questions on NumPy Sorting and Searching
#How to get the indices of the sorted array using NumPy in Python?
#Finding the k smallest values of a NumPy array
#How to get the n-largest values of an array using NumPy?
#Sort the values in a matrix
#Filter out integers from float numpy array
#Find the indices into a sorted array



#Questions on NumPy Mathematics
#How to get element-wise true division of an array using Numpy?
#How to calculate the element-wise absolute value of NumPy array?
#Compute the negative of the NumPy array
#Multiply 2d numpy array corresponding to 1d array
#Computes the inner product of two arrays
#Compute the nth percentile of the NumPy array
##Calculate the n-th order discrete difference along the given axis
#Calculate the sum of all columns in a 2D NumPy array
#Calculate average values of two given NumPy arrays
#How to compute numerical negative value for all elements in a given NumPy array?
#How to get the floor, ceiling and truncated values of the elements of a numpy array?
#How to round elements of the NumPy array to the nearest integer?
#Find the round off the values of the given matrix
#Determine the positive square-root of an array
#Evaluate Einstein’s summation convention of two multidimensional NumPy arrays



#Questions on NumPy Statistics
#Compute the median of the flattened NumPy array
#Find Mean of a List of Numpy Array
#Calculate the mean of array ignoring the NaN value
#Get the mean value from given matrix
#Compute the variance of the NumPy array
#Compute the standard deviation of the NumPy array
#Compute pearson product-moment correlation coefficients of two given NumPy arrays
#Calculate the mean across dimension in a 2D NumPy array
#Calculate the average, variance and standard deviation in Python using NumPy
#Describe a NumPy Array in Python



#Questions on Polynomial
#Define a polynomial function
#How to add one polynomial to another using NumPy in Python?
#How to subtract one polynomial to another using NumPy in Python?
#How to multiply a polynomial to another using NumPy in Python?
#How to divide a polynomial to another using NumPy in Python?
#Find the roots of the polynomials using NumPy
#Evaluate a 2-D polynomial series on the Cartesian product
#Evaluate a 3-D polynomial series on the Cartesian product



#Questions on NumPy Strings
#Repeat all the elements of a NumPy array of strings
#How to split the element of a given NumPy array with spaces?
#How to insert a space between characters of all the elements of a given NumPy array?
#Find the length of each string element in the Numpy array
#Swap the case of an array of string
#Change the case to uppercase of elements of an array
#Change the case to lowercase of elements of an array
#Join String by a seperator
#Check if two same shaped string arrayss one by one
#Count the number of substrings in an array
#Find the lowest index of the substring in an array
#Get the boolean array when values end with a particular character
#More Questions on NumPy
#Different ways to convert a Python dictionary to a NumPy array
#How to convert a list and tuple into NumPy arrays?
#Ways to convert array of strings to array of floats
#Convert a NumPy array into a csv file
#How to Convert an image to NumPy array and save it to CSV file using Python?
#How to save a NumPy array to a text file?

