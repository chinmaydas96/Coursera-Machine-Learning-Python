def linearKernel(x1, x2):
    """returns a linear kernel between x1 and x2
    and returns the value in sim
    """

# Ensure that x1 and x2 are column vectors
    x1 = x1.ravel()
    x2 = x2.ravel()

# Compute the kernel
    sim = x1.T.dot(x2)  # dot product

    return sim
