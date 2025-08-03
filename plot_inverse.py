import numpy as np  # for numerical operations and complex number support
import matplotlib.pyplot as plt  # for plotting figures
from matplotlib.lines import Line2D  # for custom legend lines
import matplotlib.ticker as ticker  # for setting axis tick spacing and formatting

# compute the n-ellipse: sum of distances from each point Z to all foci
def compute_n_ellipse(Z, foci):
    dist_sum = np.zeros_like(Z, dtype=np.float64)  # initialize array for sum of distances
    for f in foci:
        dist_sum += np.abs(Z - f)  # add distance from each point Z to the focus f
    return dist_sum  # return the total distance at each grid point

# compute the Möbius-transformed version: apply 1/z, then compute distances to foci
def compute_transformed_curve(Z, foci):
    epsilon = 1e-6  # small value to avoid division by zero
    # replace values close to zero with NaN to avoid invalid division
    Z_safe = np.where(np.abs(Z) < epsilon, np.nan + 1j*np.nan, Z)
    dist_sum = np.zeros_like(Z, dtype=np.float64)  # initialize sum of distances
    for f in foci:
        dist_sum += np.abs(1/Z_safe - f)  # add distance from transformed point to focus f
    return dist_sum  # return the total distance at each transformed point

# main function to plot both original and transformed curves
def plot_two_curves(foci, r, xlim=(-6,6), ylim=(-6,6), resolution=1000):
    # create grid of complex numbers over the given x and y range
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y  # convert to complex numbers

    # compute distance sum for the original n-ellipse and transformed curve
    dist_n_ellipse = compute_n_ellipse(Z, foci)
    dist_transformed = compute_transformed_curve(Z, foci)

    # create the plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # plot level curve of the n-ellipse where distance sum = r
    ax.contour(X, Y, dist_n_ellipse, levels=[r], colors='blue', linewidths=2, linestyles='solid')
    # plot level curve of the transformed shape with same r
    ax.contour(X, Y, dist_transformed, levels=[r], colors='#f71919', linewidths=2, linestyles='dashed')

    # plot the foci as 'x' markers
    fx = [f.real for f in foci]
    fy = [f.imag for f in foci]
    ax.scatter(fx, fy, color='#cc99ff', marker='x', s=50, label='Foci')

    # set axis labels
    ax.set_xlabel("Re(z)", loc='right')  # real part label
    ax.set_ylabel("Im(z)", loc='top')    # imaginary part label
    ax.set_aspect('equal') 
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # draw axis lines at x=0 and y=0
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)

    # hide the default four border lines of the plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(np.arange(-6, 7, 2))
    ax.set_yticks(np.arange(-6, 7, 2))

    ax.tick_params(axis='x', which='major', direction='inout', length=5, pad=2)
    ax.tick_params(axis='y', which='major', direction='inout', length=5, pad=2)

    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    plt.xticks(fontsize=10, color='black')
    plt.yticks(fontsize=10, color='black')

    # add a legend with line and marker descriptions
    ax.legend(handles=[
        Line2D([0], [0], color='blue', linestyle='-', linewidth=2, label='n-ellipse'),
        Line2D([0], [0], color="#f71919", linestyle='--', linewidth=2, label='Transformed curve'),
        Line2D([0], [0], color='#cc99ff', marker='x', linestyle='None', markersize=8, label='Foci'),
    ])

    # show the plot window
    plt.show()

# example
foci = [-0.55+0.73j, 5.7-0.65j, -3.23+1.22j, 4.9-3.24j]
r = 17
plot_two_curves(foci, r)
