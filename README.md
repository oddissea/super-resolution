# Towards spectral super-resolution in solar absorption lines from pseudomonochromatic images


## Software installation process documentation

### Introduction

This documentation guides the installation and setup of the necessary environment for the super-resolution project using Conda.

### Prerequisites

- Access to a terminal or command line
- Internet connection to download necessary packages
- Administrative permissions (if necessary to install Conda)

### Step 1: Installing Conda

#### Download and install Miniconda or Anaconda

- Visit the [official Miniconda site](https://docs.conda.io/en/latest/miniconda.html) or the [official Anaconda site](https://www.anaconda.com/products/individual) and download the appropriate installer for your operating system.
- Follow the instructions to complete the installation.

#### Verify the installation

- Restart your terminal and type:
```bash
conda --version
```
### Step 2: Setting up the Conda environment

#### Navigate to the project folder

- Change to the directory where the project is located, if you are not already there:

```bash
cd path/to/FHN
```

#### Create the environment from a super-resolution.yml file

- Ensure the super-resolution.yml file is in your current directory or specify the full path to the file.
- Run the following command:

```bash
conda env create -f super-resolution.yml
```

- This will create a new environment named super-resolution and install all the dependencies specified in the file.

#### Activate the environment

- Once the environment is created, activate it with:

```bash
conda activate super-resolution
```

### Step 3: Working with the project

#### Start JupyterLab

- Run JupyterLab with:

```bash
jupyter lab
```

This will open JupyterLab in your browser, where you can open and run the project notebooks.

### Opening PyTorch Profiler logs on TensorBoard

Assuming `tensorboard` and `torch-tb-profiler` are installed, follow these steps to view your profiling logs:

1. **Check the location of TensorFlow**
   - Run the following command to find out where TensorFlow is installed:
     ```bash
     pip show tensorflow
     ```
   - This will output information including the location of TensorFlow, for example:
     ```
     Name: tensorflow
     Version: 1.4.0
     Location: /home/abc/xy/.local/lib/python2.7/site-packages
     ```

2. **Navigate to the TensorFlow directory**
   - Go to the directory you obtained from the above output:
     ```bash
     cd /home/abc/xy/.local/lib/python2.7/site-packages
     ```
   - Inside this directory, you should find a folder named `tensorboard`:
     ```bash
     cd tensorboard
     ```

3. **Locate the Main Python File**
   - There should be a file named `main.py` in the `tensorboard` directory.

4. **Launch TensorBoard**
   - Execute the following command to start TensorBoard and view the profiler logs:
     ```bash
     python main.py --logdir=/path/to/log_file/
     ```
   - Alternatively, if you are using Python 3:
     ```bash
     python3 main.py --logdir=/path/to/log_file/
     ```

These steps will allow you to view the profiler logs on TensorBoard, providing insights into the performance of your PyTorch models.