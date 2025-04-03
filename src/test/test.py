import numpy as np

# Q1: Gaussian Elimination
# Define the augmented matrix
# Create a random 3x4 matrix with integer values between 1 and 9
np.random.seed(0)  # Set seed for reproducibility
A = np.random.randint(1, 10, size=(3, 4)).astype(float)
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
np.random.seed(1)  # Set seed for reproducibility
B = np.random.randint(1, 10, size=(4, 4)).astype(float)

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
np.random.seed(2)  # Set seed for reproducibility
C = np.random.randint(1, 10, size=(5, 5)).astype(float)

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
np.random.seed(0)  # Set seed for reproducibility
D = np.random.randint(1, 10, size=(3, 3)).astype(float)

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