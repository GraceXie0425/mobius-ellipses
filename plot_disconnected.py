import numpy as np  # numerical operations, complex support
import matplotlib.pyplot as plt  # plotting

def compute_n_ellipse(Z, foci):
    dist_sum = np.zeros_like(Z, dtype=np.float64)
    for f in foci:
        dist_sum += np.abs(Z - f)
    return dist_sum

def mobius_transform(z):
    return (3*z + 2) / (z - 0.5)

def extract_contour_points(X, Y, dist_sum, r):
    cs = plt.contour(X, Y, dist_sum, levels=[r])
    paths = cs.collections[0].get_paths()
    plt.close()  # prevent display of the contour plot itself
    return [path.vertices for path in paths]

def plot_combined_n_ellipse_and_mobius(foci, r, xlim=(-6,6), ylim=(-6,6), resolution=1000):
    # generate grid over complex plane
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    # compute original n-ellipse distances and extract contour points
    dist_original = compute_n_ellipse(Z, foci)
    curve_original = extract_contour_points(X, Y, dist_original, r)
    
    # compute transformed points by applying Möbius transform
    curve_transformed = []
    for p in curve_original:
        transformed = mobius_transform(p[:,0] + 1j * p[:,1])
        mask = np.abs(transformed) < 100
        transformed = transformed[mask]
        curve_transformed.append(transformed)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # plot original n-ellipse contours (blue solid), only label first for legend
    for i, p in enumerate(curve_original):
        ax.plot(p[:,0], p[:,1], color='blue', linestyle='-', linewidth=1.5, label='n-ellipse' if i == 0 else None)
    
    # plot Möbius transformed contours (red dashed), only label first
    for i, p in enumerate(curve_transformed):
        ax.plot(p.real, p.imag, color='#f71919', linestyle='-', linewidth=1.5, label='Möbius transformed' if i == 0 else None)

    # style
    fx = [f.real for f in foci]
    fy = [f.imag for f in foci]
    ax.scatter(fx, fy, color='#cc99ff', marker='x', s=50, label='Foci')
    
    ax.set_xlabel("Re(z)", loc='right')
    ax.set_ylabel("Im(z)", loc='top')
    
    ax.set_aspect('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ticks = np.arange(xlim[0], xlim[1]+1, 2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    ax.tick_params(axis='x', which='major', direction='inout', length=5, pad=2)
    ax.tick_params(axis='y', which='major', direction='inout', length=5, pad=2)
    
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    
    plt.xticks(fontsize=10, color='black')
    plt.yticks(fontsize=10, color='black')
    
    ax.legend(loc='upper right', fontsize=10)
    
    plt.show()

# example
if __name__ == "__main__":
    foci = np.array([4 + 0j, -4 + 1j, -3 + 5j])
    r = 18
    plot_combined_n_ellipse_and_mobius(foci, r)
