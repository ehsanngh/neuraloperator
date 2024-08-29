import matplotlib.pyplot as plt
        
def plot_2D(x, y, title=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    if x.dim() == 4:
        cp = ax.contourf(x[0, 0], x[0, 1], y[0, 0], levels=20)
    elif x.dim() == 2:
        cp = ax.tricontourf(x[:, 0], x[:, 1], y[:, 0], levels=20)
    else:
        raise ValueError(f"Unsupported dimension for x.")
    fig.subplots_adjust(right=0.9)
    ax.set_title(title)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    fig.colorbar(cp, cax=cbar_ax)
    fig.show()
