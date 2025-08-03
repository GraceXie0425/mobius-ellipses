import numpy as np  # numerical operations, complex support
import matplotlib.pyplot as plt  # plotting
from matplotlib.lines import Line2D  # for custom legend lines

def compute_n_ellipse(Z, foci):
    """Compute sum of distances from each point in Z to all foci (n-ellipse)."""
    dist_sum = np.zeros_like(Z, dtype=np.float64)
    for f in foci:
        dist_sum += np.abs(Z - f)
    return dist_sum

def mobius_transform(z):
    z_safe = np.where(z == -1, -0.999999 + 0j, z)
    return (z_safe - 1) / (z_safe + 1)

def extract_contour_points(X, Y, dist_sum, r):
    cs = plt.contour(X, Y, dist_sum, levels=[r])
    paths = cs.collections[0].get_paths()
    plt.close()  # prevent display
    return [path.vertices for path in paths]

def plot_combined_n_ellipse_and_mobius(foci, r, xlim=(-6,6), ylim=(-6,6), resolution=1000):
    
    # Generate grid over complex plane
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    # Compute original n-ellipse distances and contour points
    dist_original = compute_n_ellipse(Z, foci)
    curve_original = extract_contour_points(X, Y, dist_original, r)
    
    # Compute transformed points by applying Möbius transform to original contour points
    curve_transformed = [mobius_transform(p[:,0] + 1j*p[:,1]) for p in curve_original]
    
    # Plotting setup
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot original n-ellipse contours in blue solid lines
    for p in curve_original:
        ax.plot(p[:,0], p[:,1], color='blue', linestyle='-', linewidth=2)
    
    # Plot Möbius transformed contours in red dashed lines
    for p in curve_transformed:
        ax.plot(p.real, p.imag, color='#f71919', linestyle='--', linewidth=2)
    
    # Plot foci as purple 'x' markers
    fx = [f.real for f in foci]
    fy = [f.imag for f in foci]
    ax.scatter(fx, fy, color='#cc99ff', marker='x', s=50, label='Foci')
    
    # Axis labels with location adjustment
    ax.set_xlabel("Re(z)", loc='right')
    ax.set_ylabel("Im(z)", loc='top')
    
    # Axis equal aspect and limits
    ax.set_aspect('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Draw axis lines at x=0 and y=0 (origin axes)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    
    # Hide default spines (border lines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Custom ticks every 2 units
    ticks = np.arange(xlim[0], xlim[1]+1, 2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    # Ticks style: inward-outward, length and padding
    ax.tick_params(axis='x', which='major', direction='inout', length=5, pad=2)
    ax.tick_params(axis='y', which='major', direction='inout', length=5, pad=2)
    
    # Move bottom and left spines to zero (origin)
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    
    # Font size and color for ticks
    plt.xticks(fontsize=10, color='black')
    plt.yticks(fontsize=10, color='black')
    
    # Legend with line and marker explanations
    ax.legend(handles=[
        Line2D([0], [0], color='blue', linestyle='-', linewidth=2, label='n-ellipse'),
        Line2D([0], [0], color="#f71919", linestyle='--', linewidth=2, label='Möbius transformed'),
        Line2D([0], [0], color='#cc99ff', marker='x', linestyle='None', markersize=8, label='Foci'),
    ])
    
    plt.show()

# Example usage
if __name__ == "__main__":
    foci = np.array([2 + 0j, 1 + 1j, -1 + 2j])
    r = 7
    plot_combined_n_ellipse_and_mobius(foci, r)