# Import the NumPy library, commonly used for numerical operations,
# and rename it as np for convenience.
import numpy as np

# Import the 'random' library to generate random numbers
# and perform random operations.
import random

# Import the pyplot module from the Matplotlib library,
# commonly used for plotting and data visualization,
# and rename it as plt for convenience.
import matplotlib.pyplot as plt

# Import Poly3DCollection from mpl_toolkits.mplot3d.art3d, a Matplotlib tool 
# for handling 3D polygon collections, useful for advanced visualization 
# in three-dimensional plots.
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Import LinearSegmentedColormap from matplotlib.colors, a class in Matplotlib 
# used for creating custom color maps. This allows for greater flexibility in 
# data visualization by enabling specific colors to be defined for different 
# value ranges in a plot.
from matplotlib.colors import LinearSegmentedColormap

# Import the 'seaborn' library for data visualization. Seaborn is based on Matplotlib
# and provides a high-level interface for drawing attractive and informative statistical graphics.
import seaborn

# Import the h5py library, which is used for reading and writing HDF5 files,
# a format to store large datasets.
import h5py

# Import the psutil library, which is used for retrieving information
# on system utilization (CPU, memory, disks, etc.)
import psutil

# Import the 'platform' module to detect the operating system type
import platform

# Import the subprocess library for interacting with the operating system.
# This library allows you to execute shell commands, get their output, etc.
import subprocess

# Import the PyTorch library, a popular deep learning framework.
import torch

# Import the 'transforms' module from PyTorch's torchvision library,
# which provides common image transformations.
import torchvision.transforms as T

# Import the 'Dataset' class from PyTorch's utility data module,
# which provides an abstract class for custom datasets.
from torch.utils.data import Dataset


# Define a function to set the random seed for reproducibility across various parts of the code.
def set_seed(seed):
    # Set the seed for generating random numbers for PyTorch.
    torch.manual_seed(seed)

    # Set the seed for generating random numbers for all GPUs.
    torch.cuda.manual_seed_all(seed)

    # Ensures that the same sequence of random numbers is generated every time.
    torch.backends.cudnn.deterministic = True

    # Disables the cudnn auto-tuner for reproducibility.
    torch.backends.cudnn.benchmark = False

    # Set the seed for NumPy's random number generator.
    np.random.seed(seed)

    # Set the seed for Python's built-in random library.
    random.seed(seed)


def get_data(path):
    """
    Define a function called 'get_data' that takes a file path as an argument.
    """

    # Open the HDF5 file at the given path in read-only mode ('r').
    with h5py.File(path, 'r') as f:
        # Get the first key from the HDF5 file's keys and store it in 'primera_clave'.
        primera_clave = list(f.keys())[0]

        # Print the first key for debugging or informational purposes.
        print(f'First key: {primera_clave}')

        # Use the first key to access the corresponding dataset and return its content.
        # In this case, the key is expected to be 'cube'.
        return f[primera_clave][:]


def get_experiment_data(dataset, index=None):
    """
    Define a function called 'get_experiment_data' that takes a 3D dataset
    and an optional list of indices as arguments.

    Parameters:
    dataset (numpy.ndarray): A 3D array containing images of the Sun's surface (shape: (21, 966, 964)).
    index (list, optional): List of indices of the images to select along the λ axis.
    If not provided, the complete tensor is returned.

    Returns:
    tuple: (torch.Tensor, str) if individual images are selected,
    (torch.Tensor, list) if the complete tensor is returned.
    """

    # Check if an index list is provided.
    if index is not None:
        # Check if all indices are within the valid range.
        if all(0 <= i < dataset.shape[0] for i in index):
            # Select and return the images at the positions specified by 'index' along the λ axis.
            selected_images = dataset[index, :, :]
            return torch.tensor(selected_images, dtype=torch.float32).unsqueeze(1)
        else:
            # Raise a ValueError if any index is out of range.
            raise ValueError(f'One or more indices are out of range. Please choose indices between 0 and {dataset.shape[0] - 1}.')
    else:
        # Return the complete 'cube' tensor in PyTorch and 'None' for the wavelength label.
        return torch.tensor(dataset, dtype=torch.float32).unsqueeze(1)


def crop(dataset, crop_size=None):
    """
    Define a function called 'crop' that takes a dataset
    and an optional crop_size as arguments.
    """

    # Check if a crop_size is provided.
    if crop_size is not None:
        # Initialize the CenterCrop operation from
        # PyTorch's torchvision.transforms module.
        center_crop = T.CenterCrop(crop_size)

        # Apply the CenterCrop operation to each image in the tensor
        # and stack them back into a new tensor.
        return torch.stack([center_crop(data) for data in dataset])

    # If no crop_size is provided, return the original dataset.
    return dataset


def normalize(tensor):
    """
    Define a function called 'normalize' that takes a tensor as an argument.
    """
    # Find the minimum value in the tensor.
    min_val = torch.min(tensor)

    # Find the maximum value in the tensor.
    max_val = torch.max(tensor)

    # Normalize the tensor to the range [0, 1] using the formula: (tensor - min) / (max - min).
    normalized_tensor = (tensor - min_val) / (max_val - min_val)

    # Return the normalized tensor along with the original minimum and maximum values.
    return normalized_tensor, (min_val, max_val)


def denormalize(normalized_tensor, min_value, max_value):
    """
    Denormalize a tensor from the [0, 1] interval to its original scale using the global minimum and maximum values.
    Args:
        normalized_tensor: A normalized PyTorch tensor.
        min_value: The global minimum value used for normalizing.
        max_value: The global maximum value used for normalizing.
    Returns:
        A denormalized tensor.
    """

    # Denormalize the tensor using the original global min-max values.
    denormalized_tensor = normalized_tensor * (max_value - min_value) + min_value

    # Return the denormalized tensor.
    return denormalized_tensor


def feat_scaling(tensor):
    """
    Define a function called 'feat_scaling' that takes a PyTorch tensor as an argument.

    Normalizes a tensor along the last two axes using min-max scaling.
    Args:
        tensor: A PyTorch tensor.
    Returns:
        A normalized tensor and a tuple of minimum and maximum values
        used in the normalization.
    """

    # Calculate the minimum values of the tensor along the last two axes.
    min_values = tensor.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]

    # Calculate the maximum values of the tensor along the last two axes.
    max_values = tensor.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

    # Normalize the tensor using min-max scaling.
    normalized_tensor = (tensor - min_values) / (max_values - min_values)

    # Return the normalized tensor and a tuple of minimum and maximum values
    # used in the normalization.
    return normalized_tensor, (min_values, max_values)


def feat_descale(normalized_tensor, min_max_values):
    """
    Define a function called 'feat_descale' that takes a normalized PyTorch tensor and
    a tuple containing the minimum and maximum values used for normalization.

    Denormalizes a tensor to its original scale.
    Args:
        normalized_tensor: A PyTorch tensor that has been normalized.
        min_max_values: A tuple containing the minimum and maximum values used for normalization.
    Returns:
        A denormalized tensor with the original scale.
    """
    
    # Extract the minimum and maximum values from the tuple
    min_values, max_values = min_max_values

    # Denormalize the tensor using the min-max values
    denormalized_tensor = (normalized_tensor * (max_values - min_values)) + min_values

    return denormalized_tensor


def plot_data(dataset, labels=None, cmap=None, num_columns=7, wspace=0.3, hspace=0.3,
              tick_size=14, title_size=16, axis_size=16, title=None, path=None, img_name=None):
    """
    Define a function called 'plot_data' to visualize a dataset of images.
    The function takes several optional parameters including labels, colormap,
    number of columns, and spacing between subplots.
    """

    # Check if the dataset is a 2D array. If so, convert it to a list of one element.
    if len(dataset.shape) == 2:
        dataset = [dataset]

    # Calculate the number of images and the number of rows and columns for the grid layout.
    num_img = len(dataset)
    n_rows, n_cols = divmod(num_img, num_columns)
    n_rows += 1 if n_cols != 0 else 0
    n_cols = min(num_img, num_columns)

    # Create a grid of subplots with the calculated number of rows and columns.
    fig, axes = plt.subplots(nrows=n_rows,
                             ncols=n_cols,
                             figsize=(3 * n_cols, 3 * n_rows))

    # Initialize lists to store maximum and minimum values of images.
    max_values = []
    min_values = []

    # Loop through each image in the dataset and plot it on the grid.
    for i, img in enumerate(dataset):
        row = i // n_cols
        col = i % n_cols

        # Handle the case where there's only one subplot.
        if num_img == 1:
            axes = np.array([axes])

        # Handle the case where there's only one row of subplots.
        if len(axes.shape) == 1:
            # Configure tick parameters and plot the image.
            axes[i].tick_params(axis='both', labelsize=tick_size)
            axes[i].imshow(img, cmap=cmap)

            # Set the title based on whether labels are provided.
            if labels is not None:
                axes[i].set_title(labels[i], fontsize=title_size)
            else:
                axes[i].set_title(f'Image {i + 1}', fontsize=title_size)

            # Set x-axis labels.
            axes[i].set_xlabel('X (pixel)', fontsize=axis_size)
            xticks = [0, img.shape[1] // 4, img.shape[1] // 2, 3 * img.shape[1] // 4, img.shape[1] - 1]
            axes[i].set_xticks(xticks)

            # Set y-axis labels only if it's in the first column.
            if i % n_cols == 0:
                axes[i].set_ylabel('Y (pixel)', fontsize=axis_size)
                yticks = [0, img.shape[1] // 4, img.shape[0] // 2, 3 * img.shape[1] // 4, img.shape[0] - 1]
                axes[i].set_yticks(yticks)
            else:
                axes[i].set_yticklabels([])
        else:
            # Configure tick parameters and plot the image.
            axes[row, col].tick_params(axis='both', labelsize=tick_size)
            axes[row, col].imshow(img, cmap=cmap)

            # Set the title based on whether labels are provided.
            if labels is not None:
                axes[row, col].set_title(labels[i], fontsize=title_size)
            else:
                axes[row, col].set_title(f'Imagen {i + 1}', fontsize=title_size)

            # Set x-axis labels only in the last row.
            if row == n_rows - 1:
                axes[row, col].set_xlabel('X (pixel)', fontsize=axis_size)
                xticks = [0, img.shape[1] // 4, img.shape[1] // 2, 3 * img.shape[1] // 4, img.shape[1] - 1]
                axes[row, col].set_xticks(xticks)
            else:
                axes[row, col].set_xticklabels([])

            # Set y-axis labels only if it's in the first column.
            if col == 0:
                axes[row, col].set_ylabel('Y (pixel)', fontsize=axis_size)
                yticks = [0, img.shape[1] // 4, img.shape[0] // 2, 3 * img.shape[1] // 4, img.shape[0] - 1]
                axes[row, col].set_yticks(yticks)
            else:
                axes[row, col].set_yticklabels([])

        # Append maximum and minimum values of the image to the lists.
        max_values.append(np.max(img))
        min_values.append(np.min(img))

    # Hide the axes of the empty subplots, if any.
    for i in range(num_img, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if num_img == 1:
            axes = np.array([axes])
        if len(axes.shape) == 1:
            axes[i].axis('off')
        else:
            axes[row, col].axis('off')

    # Adjust the spacing between the subplots.
    if n_rows == 1:
        plt.tight_layout()
        # print(n_rows)

    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    # Save the plot to a PNG file
    if path is not None:
      plt.savefig(f'{path}{img_name}.png', dpi=150)

    plt.show()

    # Uncomment the following line to print the dynamic range of the images.
    # print(f'{s}\nDynamic range: ({min(min_values)}, {max(max_values)})\n')


def time_formatter(seconds):
    """
    Convert a given time in seconds to a string representation in
    hours, minutes, and seconds.

    Parameters:
        seconds (float): The total time in seconds.

    Returns:
        str: A string representing the time in hours, minutes, and seconds.
    """

    # Calculate the number of hours
    hours = seconds // 3600

    # Calculate the remaining minutes
    minutes = (seconds % 3600) // 60

    # Calculate the remaining seconds
    remaining_seconds = seconds % 60

    # Initialize an empty string to store the result
    result = ""

    # Add hours to the result if applicable
    if hours > 0:
        result += f"{hours} hour{'s' if hours > 1 else ''}"

    # Add minutes to the result if applicable
    if minutes > 0:
        result += f" {minutes} minute{'s' if minutes > 1 else ''}"

    # Add remaining seconds to the result if applicable, or if the result is still empty
    if remaining_seconds > 0 or not result:
        result += f" {remaining_seconds:.2f} second{'s' if remaining_seconds > 1 else ''}"

    # Return the result string, removing any leading or trailing whitespace
    return result.strip()


def smooth_curve(points, factor=.9):
    """
    Define a function to smooth a curve represented by a list of points.
    This is often used to smooth the learning curves during training.
    The 'factor' parameter controls the degree of smoothing.
    """

    # Initialize an empty list to store the smoothed points.
    smoothed_points = []
    # Loop through each point in the original list of points.
    for point in points:
        # If there are already points in the smoothed list,
        # apply the smoothing formula.
        if smoothed_points:
            # Get the last smoothed point.
            previous = smoothed_points[-1]
            # Calculate the new smoothed point using the factor.
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            # If this is the first point, add it to the smoothed list as is.
            smoothed_points.append(point)
    # Return the list of smoothed points.
    return smoothed_points


def plot_metrics(epochs=2000, dics={},
                 ylimits=[(10, 42), (10, 42), (-0.001, 0.1), (-0.001, 0.1)],
                 metrics=['psnr', 'val_psnr', 'loss', 'val_loss'],
                 y_labels=['TRAIN PSNR', 'TEST PSNR', 'TRAIN Loss', 'TEST Loss'],
                 filas=2, columnas=2, w=16, h=12, smooth_value=0.9, path=None, img_name=None,
                 axis_font_size=10, title_font_size=14, 
                 legend_font_size=12, tick_font_size=10):
    # from seaborn.relational import lineplot
    # Set the Seaborn theme for better visualization
    seaborn.set()
    seaborn.set_theme()

    # Create a figure and axes with two rows and two columns
    fig, axs = plt.subplots(filas, columnas, figsize=(w, h))

    # Generate x-values based on the number of epochs
    xs = np.arange(1, epochs + 1)

    # Define metrics and their corresponding y-axis labels
    metrics = metrics
    y_labels = y_labels

    # Define y-axis limits for PSNR and Loss plots
    y_limits = ylimits

    # Loop through each subplot to plot the data
    for i, ax in enumerate(axs.flatten()):
        metric = metrics[i]  # Get the metric to be plotted for this subplot
        # Set the line style based on the subplot index (even or odd)
        if (i % 2 == 0):
            line = '--'  # Dashed line for even-indexed subplots
        else:
            line = '-'  # Solid line for odd-indexed subplots

        y_label = y_labels[i]  # Get the y-axis label for this subplot
        y_min, y_max = y_limits[i]  # Get the y-axis limits for this subplot

        # Plot the smoothed curve for each dictionary in 'dics'
        for k in dics:
            curve = smooth_curve(dics[k][metric], smooth_value)
            ax.plot(xs, curve, label=k, linestyle=line)

        # Set subplot title, labels, and legend
        ax.set_title('', fontsize=title_font_size)  # <-- Añade el tamaño de la fuente aquí
        ax.set_ylabel(y_label, fontsize=axis_font_size)  # <-- Añade el tamaño de la fuente aquí
        ax.set_xlabel('Epochs', fontsize=axis_font_size)  # <-- Añade el tamaño de la fuente aquí
        ax.legend(loc='upper left', fontsize=legend_font_size) 

        # Set the font size for the ticks
        ax.tick_params(axis='both', which='major', labelsize=tick_font_size)

        # Adjust the axis limits
        ax.set_xlim(left=0)
        ax.set_xlim(right=2000)
        ax.set_ylim(bottom=0)

        # Adjust the position of the axes to start at the origin
        ax.spines['left'].set_position('zero')
        # ax.spines['bottom'].set_position('zero')

        # Remove the top and right spines for better visualization
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Set the y-axis limits based on the predefined values
        ax.set_ylim(y_min, y_max)

    # Adjust the spacing between the plots to avoid overlap
    plt.tight_layout()

    # Save the plot to a PNG file
    if path is not None:
      plt.savefig(f'{path}{img_name}.png', dpi=300)

    # Show the plots
    plt.show()

    # Reset the plot settings to default
    plt.rcdefaults()

    return plt


def get_info(command):
    # Determine the type of operating system
    os_type = platform.system()

    # Use the Jupyter syntax to capture the command output
    if os_type == "Linux":
        # output = !lscpu
        output = subprocess.check_output(["lscpu"]).decode("utf-8").splitlines()
    elif os_type == "Windows":
        # output = !wmic cpu get name
        output = subprocess.check_output(["wmic", "cpu", "get", "name"]).decode("utf-8").splitlines()
    elif os_type == "Darwin":  # macOS
        # output = !sysctl -n machdep.cpu.brand_string
        output = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode("utf-8").splitlines()
    else:
        return f"Unsupported OS: {os_type}"

    # Convert the list of strings to a single string with real newlines
    info = "\n".join(output)

    return info


def info():
    # Initialize an empty string to store the system information
    info_str = ''

    # Check if CUDA is available
    if torch.cuda.is_available():
        # output = !nvidia-smi
        output = subprocess.check_output(["nvidia-smi"]).decode("utf-8").splitlines()
        info_str += "\n".join(output) + "\n"
    else:
        info_str += get_info("") + "\n"

    # Get RAM info
    ram = psutil.virtual_memory()
    info_str += f"\nTotal RAM: {ram.total // (1024 ** 3)} GB"
    info_str += f"\nAvailable RAM: {ram.available // (1024 ** 3)} GB"
    info_str += f"\nUsed RAM: {ram.used // (1024 ** 3)} GB"

    return info_str


def plot_outputs(model, x_train, x_val, o_train, o_val, l_t, l_v):
    """
    Define a function called 'plot_outputs' to visualize the outputs of a trained model.
    The function takes the model, training and validation data,
    original training and validation outputs, and their labels as arguments.
    """

    # Set the model to evaluation mode.
    model.eval()

    # Generate images using the model for the training data and convert them to NumPy arrays.
    gen_mlp = model(x_train).squeeze().detach().cpu().numpy()

    # Generate images using the model for the validation data and convert them to NumPy arrays.
    gen_mlp_val = model(x_val).squeeze().detach().cpu().numpy()

    # Plot and display the generated images for the training set.
    print('GENERATED IMAGES\n')
    plot_data(gen_mlp[:6], l_t[:6])

    # Plot and display the original images for the training set.
    print('ORIGINAL IMAGES\n')
    plot_data(o_train[:6], l_t[:6])

    # Plot and display the generated images for the validation set.
    print('GENERATED IMAGES\n')
    plot_data(gen_mlp_val, l_v)

    # Plot and display the original images for the validation set.
    print('ORIGINAL IMAGES\n')
    plot_data(o_val, l_v)


def plot_2_st(st1, st2, st1_name, st2_name):
    """
    Define a function called 'plot_2_st' to plot two statistics over epochs.
    The function takes two statistics arrays and their names as arguments.
    """

    # Create a figure with two subplots in a single row.
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Plot the first statistic curve in the first subplot.
    x_axis = np.arange(len(st1))
    axs[0].plot(x_axis, st1)
    axs[0].set_xlabel("epochs")
    axs[0].set_ylabel(st1_name)
    axs[0].set_title(f"Evolution of {st1_name} during training")
    axs[0].grid(True)

    # Plot the second statistic curve in the second subplot.
    x_axis = np.arange(len(st2))
    axs[1].plot(x_axis, st2)
    axs[1].set_xlabel("epochs")
    axs[1].set_ylabel(st2_name)
    axs[1].set_title(f"Evolution of {st2_name} during training")
    axs[1].grid(True)

    # Show the figure.
    plt.show()


def visualize_grid(grid):
    """
    Visualizes a grid of images with depth 3 using a 3D projection.

    Args:
        grid: A PyTorch tensor of shape (num_images, 3, width, height).

    Returns:
        None.
    """
    # Extract the number of images and the width and height of the images
    num_images, _, width, height = grid.shape

    # Create a figure and a 3D subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create the X and Y coordinates for the grid
    X, Y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height), indexing='ij')

    # Iterate over the images and add each plane to the figure
    for i in range(0, num_images):
        # Get the coordinates of each image
        coords = grid[i]

        # Extract the Lambda dimension
        Z = coords[2].numpy()

        # Add the plane to the figure
        ax.plot_surface(Z, Y, X, alpha=0.5)

        # Add a label for each plane at the position (0, 0, Z[0, 0])
        ax.text(Z[0, 0], 0, 0, f'{i + 1}', fontsize=7)

    # Add labels to the axes
    ax.set_xlabel('λ')
    ax.set_ylabel('H')
    ax.set_zlabel('W')

    # Show the figure
    plt.show()


def get_size(data, global_vars):
    """
    Prints the size in megabytes of a PyTorch tensor, a NumPy array, or a PyTorch Dataset.

    Args:
        data: A PyTorch tensor, a NumPy array, or a PyTorch Dataset.
        global_vars: The globals() dictionary from the Colab notebook that calls this function.

    Returns:
        None.
    """
    if isinstance(data, torch.Tensor):
        # Calculate the size of the PyTorch tensor in bytes
        size_in_bytes = data.element_size() * data.numel()

        # Convert the size to megabytes (MB)
        size_in_megabytes = size_in_bytes / (1024 * 1024)

        # Print the size in MB
        name = next(name for name, value in global_vars.items() if value is data)
        print(f'{name} size: {size_in_megabytes:.2f} MB')

    elif isinstance(data, np.ndarray):
        # Calculate the size of the NumPy array in bytes
        size_in_bytes = data.itemsize * data.size

        # Convert the size to megabytes (MB)
        size_in_megabytes = size_in_bytes / (1024 * 1024)

        # Print the size in MB
        name = next(name for name, value in global_vars.items() if value is data)
        print(f'{name} size: {size_in_megabytes:.2f} MB')

    elif isinstance(data, Dataset):
        # Calculate the size of the PyTorch Dataset in bytes
        size_in_bytes = sum(data[i][0].element_size() * data[i][0].numel() for i in range(len(data)))

        # Convert the size to megabytes (MB)
        size_in_megabytes = size_in_bytes / (1024 * 1024)

        # Print the size in MB
        name = next(name for name, value in global_vars.items() if value is data)
        print(f'{name} size: {size_in_megabytes:.2f} MB')

    else:
        print('Unsupported data type. Only PyTorch tensors, NumPy arrays, and PyTorch Datasets are accepted.')


def blue_pal():
    # Create a new color palette based on the 'Blues' colormap,
    # but modify it to be slightly darker.
    cmap = plt.cm.Blues  # Get the original 'Blues' colormap from matplotlib.
    newcolors = cmap(np.linspace(0, 1, 256))  # Generate 256 colors from the 'Blues' colormap.
    newcolors[:, :3] *= 0.7  # Darken the colors by scaling the RGB values; adjust this value to lighten or darken the palette.
    darkblue_cmap = LinearSegmentedColormap.from_list("darkblue", newcolors)  # Create a new colormap named 'darkblue' from the modified colors.
    return darkblue_cmap


def plot_cube(dataset, path=None, img_name=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set the size of the figure
    fig.set_size_inches(12, 12)  # You can change these values as needed

    # Number of images in the dataset
    num_images = dataset.shape[0]
    separation = 5  # Define the separation between plotted images

    # Loop through each image in the cube
    for i in range(num_images):
        # Define the x-position based on the image index
        x_position = i * separation + 1
        x = np.ones((dataset.shape[1], dataset.shape[2])) * x_position
        y, z = np.meshgrid(range(dataset.shape[2]), range(dataset.shape[1]))

        # Normalize the colors
        colors = plt.cm.viridis(dataset[i, :, :] / np.max(dataset))

        # Plot the surface
        ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colors, shade=False)

    ax.set_xlabel('Z (λ)', fontsize=12, labelpad=15)
    ax.set_ylabel('X (pixel)', fontsize=12, labelpad=15)

    # Ajustar la posición de la etiqueta del eje Z
    ax.set_zlabel('Y (pixel)', fontsize=12, labelpad=15)

    # Ajustar la posición del subgráfico para dar más espacio a la etiqueta del eje Z
    fig.subplots_adjust(left=0.01, bottom=0.1, top=0.9)

    # zlabel.set_position((-2.0, 3.0, 3.2))  # Ajusta la posición (x, y, z) según tus necesidades

    # Utilizar texto en lugar de set_zlabel
    # ax.text2D(0.05, 0.95, 'Y (pixel)', transform=ax.transAxes, fontsize=12)

    # Configurar los ticks del eje x para coincidir con las posiciones de las imágenes
    ax.tick_params(axis='both', labelsize=14)
    # ax.tick_params(axis='x', labelsize=12)
    x_ticks = [i * separation + 1 for i in range(num_images)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(range(1, num_images + 1))

    # Save the plot to a PNG file
    if path is not None:
      plt.savefig(f'{path}{img_name}.png', dpi=150)

    plt.show()