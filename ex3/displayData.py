import numpy as np
from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt

from show import show

def displayData(X):
    """displays 2D data
      stored in X in a nice grid. It returns the figure handle h and the
      displayed array if requested."""

# Compute rows, cols
    m, n = X.shape
    example_width = round(np.sqrt(n))
    example_height = (n / example_width)

# Compute number of items to display
    display_rows = np.floor(np.sqrt(m))
    display_cols = np.ceil(m / display_rows)

# Between images padding
    pad = 1

# Setup blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad),
                           pad + display_cols * (example_width + pad)))

# Copy each example into a patch on the display array
    curr_ex = 0
    for j in np.arange(display_rows):
        for i in np.arange(display_cols):
            if curr_ex > m:
                break
            # Get the max value of the patch
            max_val = np.max(np.abs(X[curr_ex, : ]))
            rows = [pad + j * (example_height + pad) + x for x in np.arange(example_height+1)]
            cols = [pad + i * (example_width + pad)  + x for x in np.arange(example_width+1)]
            display_array[min(rows):max(rows), min(cols):max(cols)] = X[curr_ex, :].reshape(example_height, example_width) / max_val
            curr_ex = curr_ex + 1
        if curr_ex > m:
            break

# Display Image
    display_array = display_array.astype('float32')
    plt.imshow(display_array.T)
    plt.set_cmap('gray')
# Do not show axis
    plt.axis('off')
    show()

