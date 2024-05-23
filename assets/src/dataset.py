# Import necessary modules from PyTorch
import torch
# import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


class SolarImageDataset(Dataset):
    """
    Define a class 'SolarImageDataset' that inherits 
    from PyTorch's Dataset class
    """

    def __init__(self, x_tensor, y_tensor, wavelengths):
        # Initialize the dataset with tensors for the input images (x), 
        # labels (y), and wavelengths
        self.x = x_tensor
        self.y = y_tensor
        self.wavelengths = wavelengths

    def __len__(self):
        # Return the length of the dataset
        return len(self.x)

    def __getitem__(self, idx):
        # Return a single item from the dataset given an index (idx)
        return self.x[idx], self.y[idx], self.wavelengths[idx]


class SolarFlatDataset(Dataset):
    """
    Define a custom PyTorch Dataset class for handling 
    solar data in a flattened format
    """

    def __init__(self, x_tensor, y_tensor, wavelengths=None):
        # Reorganize the tensors so that the batch dimension is at the beginning
        self.x = x_tensor.permute(2, 0, 1, 3)
        b, _, _, dim = self.x.shape  # Extract the shape of the tensor
        self.x = self.x.reshape(b, -1, dim)  # Flatten the tensor

        # Do the same for the y tensor
        self.y = y_tensor.permute(2, 0, 1, 3)
        b, _, _, dim = self.y.shape
        self.y = self.y.reshape(b, -1, dim)

        # Create a tensor 'z' that contains numbers from 1 to 'dim'
        self.z = torch.arange(1, dim + 1)

    def __len__(self):
        # Ensure that both x and y tensors have the same number of elements
        assert len(self.x) == len(self.y)
        return len(self.x)

    def __getitem__(self, index):
        # Return the corresponding elements from x, y, and z tensors for a given index
        return self.x[index], self.y[index], self.z[index]


def split_data_2D(data, labels, wavelengths, ds_type=SolarImageDataset, bs=1, device='cpu'):
    """
    This function takes in 2D data, corresponding labels, and wavelengths, and returns
    DataLoader objects for training and testing sets.

    Parameters:
        data (torch.Tensor): The input data tensor of shape (n, c, w, h).
        labels (torch.Tensor): The corresponding labels tensor of shape (n, c, w, h).
        wavelengths (list): List of wavelengths corresponding to each image.

    Returns:
        train_loader (DataLoader): DataLoader object for the training set.
        test_loader (DataLoader): DataLoader object for the testing set.
    """

    if ds_type is SolarImageDataset:
        down_data = data[:, :, ::2, ::2]
        down_labels = labels[:, :, ::2, ::2]
    else:
        down_data = data[::2, ::2, :, :]
        down_labels = labels[::2, ::2, :, :]

    # Create a dataset for training by down-sampling the data by a factor of 2.
    # The dataset is created using the 'get_dataset' function from the 'ds' module.
    train_ds = get_dataset(down_data, down_labels, wavelengths, ds_type=ds_type, device=device)

    # Split the training dataset into training and validation sets using the 'split_data' function from the 'ds' module.
    # We only keep the training set and discard the validation set.
    train_loader, _ = split_data(train_ds, batch_size=bs)

    # Create a dataset for testing using the original data.
    test_ds = get_dataset(data, labels, wavelengths, ds_type=ds_type, device=device)

    # Split the testing dataset into training and validation sets.
    # We only keep the testing set and discard the validation set.
    test_loader, _ = split_data(test_ds, batch_size=bs)

    return train_loader, test_loader


def split_data(dataset, train_ratio=1., batch_size=1):
    """
    Define a function 'split_data' to split a dataset into training and validation sets
    """

    # Print the total size of the dataset
    # print(f'The size of the dataset is {len(dataset)}')

    # Adjust the train_ratio if the dataset only has one element
    if len(dataset) == 1:
        train_ratio = 1.0

    # Calculate the validation ratio
    # val_ratio = round((1.0 - train_ratio), 1)

    # Print the training and validation ratios
    # print(f'train_ratio: {train_ratio}')
    # print(f'val_ratio: {val_ratio}')

    # Calculate the sizes for the training and validation sets
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size

    # Print the sizes for the training and validation sets
    # print(f'total: {total_size}')
    # print(f'train: {train_size}')
    # print(f'val: {val_size}')

    # If the dataset has only one element, use it for both training and validation
    if len(dataset) == 1:
        train_dataset, val_dataset = dataset, dataset
    else:
        # Split the dataset into training and validation sets
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Set the batch size for the DataLoader
    if batch_size == "None":
        batch_size = train_size
    else:
        batch_size = int(batch_size)

    # Create DataLoader objects for the training and validation sets
    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Return the DataLoader objects for the training and validation sets
    return train, val


def grid3D(data, ini_dim=0):
    """
    Define a function 'grid3D' to create a 3D grid of coordinates
    """

    # Extract the number of images, width, and height from the data tensor shape
    num_images, _, width, height = data.shape

    # Create a set of linearly spaced points in the interval [0, 1] for each dimension
    w_points = torch.linspace(ini_dim, 1, width)
    h_points = torch.linspace(ini_dim, 1, height)
    lambda_points = torch.linspace(ini_dim, 1, num_images)

    # Create the 3D grid using torch.meshgrid()
    W, H, LAMBDA = torch.meshgrid(w_points,
                                  h_points,
                                  lambda_points,
                                  indexing='ij')

    # Combine the tensors W, H, and LAMBDA into a single tensor
    coords = torch.stack((W, H, LAMBDA), dim=-1)

    # Rearrange the tensor coords so that it has the shape (num_images, 3, width, height)
    coords = coords.permute(2, 3, 0, 1)

    # Return the 3D grid of coordinates
    return coords


def get_dataset(x, y, wavelengths, device='cpu', fm=None, B=None, ds_type=SolarImageDataset):
    """
    Define a function 'get_dataset' to create a dataset object for solar images
    """

    # Move the input tensors x and y to the specified device (CPU or GPU)
    x = x.to(device)
    y = y.to(device)

    # Convert wavelengths to a tensor and move it to the device
    # Note: but you may want to uncomment it based on your needs.
    # wavelengths = torch.tensor(wavelengths).to(device)

    # Apply the feature mapping function 'fm' to x if provided, otherwise use x as is
    if fm is not None:
        ff_x = fm(x, B)
    else:
        ff_x = x

    # Create and return the complete dataset using the SolarImageDataset class
    return ds_type(ff_x, y, wavelengths)


class SingleImageDataset(Dataset):
    """
    Define another class 'SingleImageDataset' that also inherits from PyTorch's Dataset class
    """

    def __init__(self, x_tensor, y_tensor):
        # Initialize the dataset with tensors for the input images (x) and labels (y)
        self.x = x_tensor
        self.y = y_tensor

    def __len__(self):
        # Return the length of the dataset
        return len(self.x)

    def __getitem__(self, idx):
        # Return a single item from the dataset given an index (idx)
        return self.x[idx], self.y[idx]


# ATTEMPT TO CREATE PATCHES OF IMAGES
# WITHOUT LOSING THE CORRESPONDENCE IN THE COORDINATES
def create_subimage_coords(data, subimage_size=128):
    """
    Define a function 'create_subimage_coords' to create sub-images and their corresponding coordinates
    """

    # Extract the shape of the data tensor
    num_images, _, width, height = data.shape

    # Create a 3D grid of coordinates for the entire image using the 'grid3D' function
    coords = grid3D(data)

    # Initialize lists to store sub-images, their coordinates, and wavelengths
    subimages = []
    subimage_coords = []
    wavelengths = []  # List to store the wavelengths

    # Loop through each image in the dataset
    for n in range(num_images):
        # The wavelength corresponding to this image
        wavelength = n + 1

        # Initialize lists to store sub-images and their coordinates for this image
        subimages_per_image = []
        subimage_coords_per_image = []
        wavelengths_per_image = []  # Wavelengths for the portions of this image

        # Loop through the width and height to create sub-images
        for i in range(0, width, subimage_size):
            for j in range(0, height, subimage_size):
                # Append the sub-image to the list
                subimages_per_image.append(data[n, :, i:i+subimage_size, j:j+subimage_size])

                # Append the coordinates of the sub-image to the list
                subimage_coords_per_image.append(coords[n, :, i:i+subimage_size, j:j+subimage_size])

                # Add the wavelength to the list
                wavelengths_per_image.append(wavelength)

        # Append the lists to the main lists
        subimages.append(subimages_per_image)
        subimage_coords.append(subimage_coords_per_image)
        wavelengths.append(wavelengths_per_image)  # Add the wavelengths of this image to the main list

    # Return the lists of sub-images, their coordinates, and wavelengths
    return subimages, subimage_coords, wavelengths


def number_of_patches(image_width, image_height, patch_size):
    """
    This function calculates the number of patches that can fit into an image
    given the dimensions of the image and the size of the patch.

    Parameters:
    - image_width (int): The width of the image.
    - image_height (int): The height of the image.
    - patch_size (int): The size of the patch (assuming the patch is square).

    Returns:
    - num_patches (int): The number of patches that can fit into the image.
    """

    # Calculate the number of patches in each dimension
    patches_width = image_width // patch_size
    patches_height = image_height // patch_size

    # Calculate the total number of patches
    num_patches = patches_width * patches_height

    return num_patches


def optimal_patch_size(image_width, image_height, min_patch_size=64):
    """
    Finds the optimal patch size, starting from a minimum patch size, that results in the maximum
    area coverage of the image, with the area being as close as possible to the original image area.

    Parameters:
    - image_width (int): The width of the image.
    - image_height (int): The height of the image.
    - min_patch_size (int): The minimum patch size to consider.

    Returns:
    - optimal_patch_size (int): The optimal size of the patch.
    - num_patches (int): The number of patches that can fit into the image with the optimal patch size.
    """
    max_area = 0
    optimal_patch_size = min_patch_size
    image_area = image_width * image_height

    # Start from the minimum patch size and go up to the smallest dimension of the image.
    for patch_size in range(min_patch_size, min(image_width, image_height) + 1):
        num_patches_width = image_width // patch_size
        num_patches_height = image_height // patch_size
        total_patches_area = num_patches_width * num_patches_height * patch_size**2

        if total_patches_area > max_area and total_patches_area <= image_area:
            max_area = total_patches_area
            optimal_patch_size = patch_size

    # Recalculate number of patches with the optimal patch size to ensure consistency
    num_patches = (image_width // optimal_patch_size) * (image_height // optimal_patch_size)

    return optimal_patch_size, num_patches


def reconstruct_image(patches, original_image_shape, patch_size):
    """
    Reconstructs an image from its patches.

    Args:
    - patches (Tensor): A tensor containing image patches.
    - original_image_shape (tuple): The shape of the original image (C, H, W) where
      C is the number of channels, H is the height, and W is the width.
    - patch_size (int): The size of each patch. Assumes patches are square.

    Returns:
    - Tensor: The reconstructed image of shape `original_image_shape`.
    """
    
    # Initialize a tensor filled with zeros to store the reconstructed image.
    reconstructed_image = torch.zeros(original_image_shape)

    # Initialize a variable to keep track of the current patch index.
    patch_idx = 0
    
    # Iterate over the image dimensions in steps of `patch_size` to place each patch.
    for i in range(0, original_image_shape[1], patch_size):
        for j in range(0, original_image_shape[2], patch_size):
            # Determine the height and width of the patch, adjusting for the edges of the image.
            patch_height = min(patch_size, original_image_shape[1] - i)
            patch_width = min(patch_size, original_image_shape[2] - j)

            # Place the current patch into the reconstructed image tensor.
            reconstructed_image[:, i:i+patch_height, j:j+patch_width] = patches[patch_idx][:, :patch_height, :patch_width]
            
            # Move to the next patch.
            patch_idx += 1

    # Return the reconstructed image.
    return reconstructed_image


class BigSolarDataSet(Dataset):
    def __init__(self, subimages, subimage_coords):
        """
        Initialize the BigSolarDataSet class.

        Parameters:
        - subimages (list): A list containing sub-images.
        - subimage_coords (list): A list containing coordinates for each sub-image.

        The lists are "flattened" to create a single list of sub-images and coordinates.
        """

        # "Flatten" the list of subimage coordinates
        self.subimage_coords = [coords for batch in subimage_coords for coords in batch]

        # "Flatten" the list of subimages
        self.subimages = [img for batch in subimages for img in batch]

        # Create a list of wavelengths (lambdas) for each subimage
        self.lambdas = [i // len(subimages[0]) for i in range(len(self.subimages))]

    def __len__(self):
        """
        Return the total number of sub-images in the dataset.
        """
        return len(self.subimages)

    def __getitem__(self, idx):
        """
        Retrieve the sub-image, its coordinates, and its corresponding wavelength
        based on the given index.

        Parameters:
        - idx (int): The index of the sub-image to retrieve.

        Returns:
        - tuple: A tuple containing the coordinates, the sub-image, and the wavelength.
        """
        return self.subimage_coords[idx], self.subimages[idx], self.lambdas[idx]

# +++ END OF ATTEMPT +++