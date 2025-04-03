import numpy as np

# Q1: Gaussian Elimination
# Define the augmented matrix
A = np.array([
    [2, -1, 1, 6],
    [1, 3, 1, 0],
    [-1, 5, 4, -3]
], dtype=float)

def gaussian_elimination_solve(augmented_matrix):
    # Perform Gaussian elimination
    n = augmented_matrix.shape[0]
    for i in range(n):
        # Make the diagonal element 1 and eliminate below
        for j in range(i + 1, n):
            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j, i:] -= factor * augmented_matrix[i, i:]

    # Perform backward substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i + 1:n], x[i + 1:])) / augmented_matrix[i, i]

    return x

# Solve the system using the function
solution = gaussian_elimination_solve(A)

# Print the solution
print("\nSolution to this system:")
print(solution)

# Q2: LU Factorization and Determinant Calculation
# Define the matrix
B = np.array([
    [1, 1, 0, 3],
    [2, 1, -1, 1],
    [3, -1, -1, 2],
    [-1, 2, 3, -1]
], dtype=float)

# Perform LU factorization
n = B.shape[0]
L = np.eye(n)  # Initialize L as an identity matrix
U = B.copy()   # Initialize U as a copy of B

for i in range(n):
    for j in range(i + 1, n):
        factorization = U[j, i] / U[i, i]
        L[j, i] = factorization
        U[j, i:] = U[j, i:] - factorization * U[i, i:]

determinant = np.prod(np.diag(U))
determinant = np.round(determinant, 14)

# Print determinant
print("\nMatrix determinant:")
print(determinant)

# Print L 
print("\nL matrix:")
print(L)

# Print U 
print("\nU matrix:")
print(U)

# Q3: Check Diagonal Dominance
# Define the matrix
C = np.array([
    [9, 0, 5, 2, 1],
    [3, 9, 1, 2, 1],
    [0, 1, 7, 2, 3],
    [4, 2, 3, 12, 2],
    [3, 2, 4, 0, 8]
], dtype=float)

def is_diagonally_dominant(matrix):
    n = matrix.shape[0]
    for i in range(n):
        # Sum of the absolute values of the non-diagonal elements in the ith row
        sumnd = np.sum(np.abs(matrix[i])) - np.abs(matrix[i, i])
        if np.abs(matrix[i, i]) < sumnd:
            return False
    return True

# Check if the matrix is diagonally dominant
diagonally_dominant = is_diagonally_dominant(C)

# Print the result
print("\nDiagonally dominant:" if diagonally_dominant else "\nNot diagonally dominant.")

# Q4: Check Positive Definiteness
# Define the matrix
D = np.array([
    [2, 2, 1],
    [2, 3, 0],
    [1, 0, 2]
], dtype=float)

# Function to check if the matrix is positive definite
def is_pd(matrix):
    # Calculate the eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)
    # Check if all eigenvalues are positive
    return np.all(eigenvalues > 0)

# Check if the matrix is positive definite
pd = is_pd(D)

# Print the result
print("\nThe matrix is positive definite:" if pd else "The matrix is not positive definite.")