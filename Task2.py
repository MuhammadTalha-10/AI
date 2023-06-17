import numpy as np

def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# Input dataset
X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

# Output dataset
y = np.array([[0, 1, 1, 0]]).T

# Seed random numbers to make calculations deterministic (just a good practice)
np.random.seed(1)

# Initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3, 1)) - 1

for _ in range(10000):
    # Forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    # How much did we miss?
    l1_error = y - l1

    # Multiply how much we missed by the slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)

    # Update weights
    syn0 += np.dot(l0.T, l1_delta)

print("Output After Training:")
print(l1)
