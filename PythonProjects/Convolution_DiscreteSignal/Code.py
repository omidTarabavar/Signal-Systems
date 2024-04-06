import numpy as np
import matplotlib.pyplot as plt


def conv(x_n, x, h_n, h):
    """
    Convolves two discrete signals x and h.

    Args:
        x_n (numpy.ndarray): Values of n for the input signal
        x (numpy.ndarray): Input signal values
        h_n (numpy.ndarray): Values of n for the impulse response
        h (numpy.ndarray): Impulse response values

    Returns:
        numpy.ndarray: Convolved signal values
        numpy.ndarray: Corresponding values of n for the convolved signal
    """
    x_start = x_n.min()  # Starting value of n for x
    x_end = x_n.max()  # Ending value of n for x
    h_start = h_n.min()  # Starting value of n for h
    h_end = h_n.max()  # Ending value of n for h

    y_start = x_start + h_start  # Starting value of n for the output signal
    y_end = x_end + h_end  # Ending value of n for the output signal
    n_values = np.arange(y_start, y_end + 1)  # Values of n for the output signal
    y = np.zeros(len(x) + len(h) - 1, dtype= float)  # Initialize output signal
    x_index = 0
    for n in range(x_start, x_end+1):
        h_index = 0
        for k in range(h_start+n, h_end+1+n):
            if k >= n:
                a = x[x_index] * h[h_index]
                y[k] += a
            h_index += 1
        x_index += 1


    return y, n_values


# Example usage
x_n = np.array([0, 1, 2])  # Values of n for the input signal
x = np.array([2, 2, 1])  # Input signal values
h_n = np.array([0, 1, 2, 3, 4])  # Values of n for the impulse response
h = np.array([3, 3, 3, 3, 3])  # Impulse response values

result, n_values = conv(x_n, x, h_n, h)



fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8))

# Plot the first signal
ax1.stem(x_n, x)
ax1.set_title('x[n]')
ax1.set_xlabel('n')
ax1.set_ylabel('x[n]')
ax1.grid(True)

# Plot the second signal
ax2.stem(h_n, h)
ax2.set_title('h[n]')
ax2.set_xlabel('n')
ax2.set_ylabel('h[n]')
ax2.grid(True)

# Plot the third signal
ax3.stem(n_values, result)
ax3.set_title('x[n] * h[n]')
ax3.set_xlabel('n')
ax3.set_ylabel('y[n]')
ax3.grid(True)

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=1)
plt.show()