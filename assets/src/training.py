# Import the 'random' library to generate random numbers
# and perform random operations.
import random

# Import PyTorch for deep learning and tensor operations
import torch

# Import tqdm for progress bars during loops or iterations
from tqdm.auto import tqdm

# Import the 'copy' module for deep and shallow copy operations
import copy

import os

def PSNR(img1, img2):
    '''
    PSNR (Peak Signal-to-Noise Ratio)
    Arguments: img1: tensor, img2: tensor
    Returns the signal-to-noise ratio between them.
    It is a measure of the quality of the generated image.
    Higher PSNR indicates higher quality.
    '''
    # Set the maximum pixel value in the image.
    # In this case, since the images are normalized and
    # represented as floating-point tensors,
    # the maximum value is 1.0.
    # max_val = 1.0
    max_val = torch.max(img2)

    # Calculate the Mean Squared Error (MSE) between the two images.
    # MSE is the average of the squared differences
    # between corresponding pixels of the two images.
    mse = torch.mean((img1 - img2)**2)

    # Calculate the PSNR using the formula:
    # PSNR = 20 * log10(max_val / sqrt(MSE)).
    psnr_torch = 20 * torch.log10(max_val / torch.sqrt(mse))

    # Return the PSNR as a scalar value.
    return psnr_torch.detach().item()


def generate_B(m, d=3, sigma=torch.tensor([10., 10., .8], dtype=torch.float32), device='cpu'):
    """
    Generates the B matrix for Fourier mapping.

    Parameters:
        k (int): The number of Fourier projections.
        d (int): Dimension of the spatial coordinates (in this case, 3).
        sigma (float): The standard deviation for the normal distribution
                       used when sampling the Fourier bases.
        device (str): The device to which the tensor will be sent ('cpu' or 'cuda').

    Returns:
        torch.Tensor: B matrix of size (k, d).
    """
    # Generate a random tensor with shape (k, d) and multiply it by the standard deviation (sigma).
    # The tensor is of type float32 and does not require gradients.
    B = torch.randn((m, d), dtype=torch.float32, requires_grad=False) * sigma

    # Move the tensor to the specified device (either 'cpu' or 'cuda').
    return B.to(device)


def fourier_mapping(x, B):
    """
    Applies a Gaussian Fourier mapping to normalized spatial coordinates.

    Parameters:
        x (torch.Tensor): A tensor of normalized coordinates of size (n, 3, w, h).
        B (torch.Tensor): The basis matrix to project the coordinates onto.

    Returns:
        torch.Tensor: A tensor of Fourier features of size (n, 2 * k, w, h).
    """

    # Disable gradient computation for the input tensor 'x'
    x.requires_grad = False

    # Extract the shape information from the input tensor 'x'
    n, _, w, h = x.shape
    d = x.shape[1]  # Dimension of the spatial coordinates (in this case, 3)

    # Project the normalized coordinates onto the Fourier basis
    projected_coords = x.reshape(n, d, w * h).permute(0, 2, 1).matmul(B.T).reshape(n, w, h, B.shape[0])

    # Compute the cosine and sine Fourier features
    cos_features = torch.cos(2 * torch.pi * projected_coords)
    sin_features = torch.sin(2 * torch.pi * projected_coords)

    # Concatenate the cosine and sine features to form the complete Fourier feature set
    fourier_features = torch.cat((cos_features, sin_features), dim=-1)
    fourier_features = fourier_features.permute(0, 3, 1, 2)

    # Disable gradient computation for the Fourier features tensor
    fourier_features.requires_grad = False

    return fourier_features


def evaluate(model, loss_fn, loader, fm=None, B=None, device='cpu', history_images=None):
    """
    Define the evaluate function to assess the performance of the model on a given dataset.
    """

    # Set the model to evaluation mode. This disables operations like dropout.
    model.eval()

    # Disable gradient computation to save memory and speed up evaluation.
    with torch.no_grad():
        epoch_loss = 0.0  # Initialize the total loss for this epoch.
        epoch_psnr = 0.0  # Initialize the total PSNR for this epoch.
        best_psnr = -float('inf')  # Initialize the best PSNR value.
        best_generated = None  # Initialize the best generated image.
        best_val_index = None  # Initialize the index of the best validation image.
        num_batches = 0  # Initialize the number of batches processed.

        # Loop through each batch in the data loader.
        for x, y, l in loader:
            # Apply Fourier mapping if specified.
            if fm is not None:
                x = fm(x, B)

            # Generate the output using the model.
            generated = model(x)

            # Update the generated history images 
            if history_images is not None:
                history_images.append(generated)

            # Compute the loss between the generated output and the ground truth.
            loss = loss_fn(y, generated)

            # Extract the loss as a Python float.
            batch_loss = loss.item()

            # Compute the PSNR between the generated image and the ground truth.
            batch_psnr = PSNR(generated.squeeze(0), y.squeeze(0))

            # Update the best PSNR and corresponding generated image and index.
            if batch_psnr > best_psnr:
                best_psnr = batch_psnr
                best_generated = generated
                best_val_index = l

            # Update the total loss and PSNR for this epoch.
            epoch_loss += batch_loss
            epoch_psnr += batch_psnr
            num_batches += 1

        # Compute the average loss and PSNR for this epoch.
        if num_batches > 0:
            epoch_loss /= num_batches
            epoch_psnr /= num_batches

    # Return the average loss, average PSNR, best generated image, and its index.
    return epoch_loss, epoch_psnr, best_generated, best_val_index


def train_one_epoch(model, optimizer, loss_fn, loader,
                    fm=None, B=None, device='cpu', scheduler=None):
    """
    Define the function to train the model for one epoch.
    """

    # Initialize variables to keep track of the epoch's loss and PSNR.
    epoch_loss = 0.0
    epoch_psnr = 0.0
    best_psnr = -float('inf')  # Initialize the best PSNR value.
    best_generated = None  # Initialize the best generated image.
    best_train_index = None  # Initialize the index of the best training image.
    lr = None  # Initialize the learning rate.
    num_batches = 0  # Initialize the number of batches processed.

    # Set the model to training mode. This enables operations like dropout.
    model.train()

    # Loop through each batch in the data loader.
    for x, y, l in loader:
        # Zero the gradients of the model parameters.
        optimizer.zero_grad()

        # Apply Fourier mapping if specified.
        if fm is not None:
            x = fm(x, B)

        # Generate the output using the model.
        generated = model(x)

        # Compute the loss between the generated output and the ground truth.
        loss = loss_fn(y, generated)

        # Backpropagate the loss to compute gradients.
        loss.backward()

        # Update the model parameters.
        optimizer.step()

        # Update the learning rate if a scheduler is provided.
        if scheduler:
            scheduler.step()

        # Extract the loss as a Python float.
        batch_loss = loss.item()

        # Compute the PSNR between the generated image and the ground truth.
        batch_psnr = PSNR(generated.squeeze(0), y.squeeze(0))

        # Update the best PSNR and corresponding generated image and index.
        if batch_psnr > best_psnr:
            best_psnr = batch_psnr
            best_generated = generated
            best_train_index = l

        # Update the total loss and PSNR for this epoch.
        epoch_loss += batch_loss
        epoch_psnr += batch_psnr
        lr = optimizer.param_groups[0]["lr"]
        num_batches += 1

    # Compute the average loss and PSNR for this epoch.
    epoch_loss /= num_batches
    epoch_psnr /= num_batches

    # Return the average loss, average PSNR, best generated image, its index, and the learning rate.
    return epoch_loss, epoch_psnr, best_generated, best_train_index, lr


def train_epoch(model, optimizer, loss_fn, train_loader, test_loader, 
                fm, B, device, scheduler, history_images):
    # Train the model for one epoch and get the training metrics.
    epoch_loss, epoch_psnr, best_train_generated, best_train_index, lr = train_one_epoch(
        model, optimizer, loss_fn, train_loader, fm, B, device, scheduler)

    # Evaluate the model on the validation set and get the validation metrics.
    test_loss, test_psnr, best_test_generated, best_test_index = evaluate(
        model, loss_fn, test_loader, fm, B, device, history_images)
    
    return (epoch_loss, epoch_psnr, lr, test_loss, test_psnr, 
            best_train_generated, best_train_index, 
            best_test_generated, best_test_index)

def train_base(model, optimizer, loss_fn, epochs, train_loader, test_loader, 
               device='cpu', print_every=100, fm=None, B=None, scheduler=None, 
               process=False, profiler=None, base_dir=None, filename=None):
    # Initialize a dictionary to store training history.
    history = {
        'loss': [],
        'psnr': [],
        'test_loss': [],
        'test_psnr': [],
        'lr': [],
        'history_images': [],
        'best_test_psnr': -float('inf'),
        'best_epoch': -1,
        'best_train_generated': None,
        'best_train_index': None,
        'best_test_generated': None,
        'best_test_index': None
    }

    # Initialize a variable to store the best model.
    best_model = None

    # Update images 
    if process:
        history_images = history['history_images']
    else:
        history_images = None

    for epoch in tqdm(range(epochs)):
        # Call train_epoch function to reduce redundancy
        (epoch_loss, epoch_psnr, lr, test_loss, test_psnr, best_train_generated, 
         best_train_index, best_test_generated, best_test_index) = train_epoch(
            model, optimizer, loss_fn, train_loader, test_loader, 
            fm, B, device, scheduler, history_images)

        if profiler:  # If profiler is active, update it
            profiler.step()

        # Update training history.
        history['loss'].append(epoch_loss)
        history['psnr'].append(epoch_psnr)
        history['test_loss'].append(test_loss)
        history['test_psnr'].append(test_psnr)
        history['lr'].append(lr)

        # Update the best model and metrics if the validation PSNR improves.
        if test_psnr > history['best_test_psnr']:
            history['best_test_psnr'] = test_psnr
            history['best_epoch'] = epoch
            history['best_train_generated'] = best_train_generated
            history['best_train_index'] = best_train_index
            history['best_test_generated'] = best_test_generated
            history['best_test_index'] = best_test_index
            best_model = model

        # Print the metrics every 'print_every' epochs.
        if (epoch + 1) % print_every == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, test Loss: {test_loss:.4f}')
            print(f'PSNR: {epoch_psnr:.4f}, test PSNR: {test_psnr:.4f}')

        torch.cuda.empty_cache()

    return history, best_model

def train(model, optimizer, loss_fn, epochs, train_loader, test_loader, 
          device='cpu', print_every=100, fm=None, B=None, scheduler=None, 
          process=False, base_dir=None, filename=None):
    profiler = None
    if base_dir and filename:
        log_path = os.path.join(base_dir, filename)
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, 
                        torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=1, 
                                             active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler.__enter__()  # Start the profiler

    history, best_model = train_base(model, optimizer, loss_fn, epochs, 
                                     train_loader, test_loader, device, 
                                     print_every, fm, B, scheduler, process, 
                                     profiler, base_dir, filename)

    return history, best_model
    

def best_lr(model_fn, config, train_loader, test_loader, train_model,
            e=6, exp=True, min=.2, max=.4, f=3, learnig_rates={},
            B=None, device='cpu'):
    """
    Function to search for the best learning rate for a given model.

    Parameters:
    - model_fn: Function to initialize the model and optimizer.
    - config: Configuration dictionary containing model parameters.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - e (int, default=6): Exponent for the learning rate 
      when using exponential search.
    - exp (bool, default=True): Flag to determine if exponential 
      search should be used.
    - min (float, default=.2): Minimum boundary for random learning rate search.
    - max (float, default=.4): Maximum boundary for random learning rate search.
    - f (int, default=3): Factor to scale the random learning rate search interval.
    - learnig_rates (dict, default={}): Dictionary to store 
      the best PSNR values for each learning rate.
    - B: Optional matrix parameter for the model.
    - device (str, default='cpu'): Device to run the model on ('cpu' or 'cuda').

    Returns:
    - learnig_rates: Updated dictionary with the best PSNR values 
      for each tested learning rate.
    """

    # Adjust the factor for the random learning rate search interval
    f = 10**f

    # Loop through the specified number of iterations
    for i in range(e+1):
        # If exponential search is enabled
        if exp:
            # Calculate the learning rate using exponential decay
            lr = 10.0**-i
        else:
            # Calculate the learning rate using a random 
            # value within the specified interval
            min_intervalo = int(min * f)
            max_intervalo = int(max * f)
            lr = random.randrange(min_intervalo, max_intervalo)/f

        # Print the current learning rate being tested
        print(f'############ LR = {lr} ############')

        # Initialize the model and optimizer with the current learning rate
        model, optimizer = model_fn(lr, config['M'], device=device)

        # Train the model with the current learning rate
        output, best = train_model(model, optimizer, config,
                                   train_loader, test_loader, B=B)

        # Print the best PSNR value achieved with the current learning rate
        print(f"best_psnr: {output['best_test_psnr']}")

        # Store the best PSNR value for the current learning rate in the dictionary
        learnig_rates[lr] = output['best_test_psnr']

    # Return the updated dictionary with the best PSNR values for each tested learning rate
    return learnig_rates