import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def plot_field(data, geom, title, cbar_label, 
                      log_scale=False, vmin=None, vmax=None, filename=None):
    """
    Plots a 2D field with automatic masking and formatting.
    """
    # 1. Create copy and apply mask
    data_plot = np.copy(data)
    data_plot[geom.mask == 0] = np.nan
    
    # 2. Setup Figure
    plt.figure(figsize=(11, 5))
    
    # 3. Configure Normalization and Limits
    norm = None
    plot_kwargs = {} # Dictionary to hold vmin/vmax only when needed
    
    if log_scale:
        # For Log scale: vmin/vmax must go INSIDE the LogNorm object
        # We calculate defaults here if they weren't passed in
        if vmin is None: vmin = np.nanmin(data_plot)
        if vmax is None: vmax = np.nanmax(data_plot)
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        # For Linear scale: vmin/vmax go directly to pcolormesh via kwargs
        if vmin is not None: plot_kwargs['vmin'] = vmin
        if vmax is not None: plot_kwargs['vmax'] = vmax
        
    # 4. Plot
    # We unpack **plot_kwargs so we don't pass vmin/vmax if norm is present
    pcm = plt.pcolormesh(geom.Z, geom.R, data_plot, cmap='plasma', shading='auto',
                         norm=norm, **plot_kwargs)
    
    # 5. Decoration
    cbar = plt.colorbar(pcm, label=cbar_label)
    plt.xlabel('z (m)')
    plt.ylabel('r (m)')
    plt.title(title)
    plt.tight_layout()
    
    # 6. Save or Show
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved: {filename}")
    
    plt.show()