import numpy as np
import pandas as pd
import os
import SimpleITK as sitk
import cv2
import subprocess
from matplotlib import pyplot as plt
import re
import torch
from skimage.exposure import match_histograms
from medpy.filter.smoothing import anisotropic_diffusion
import nrrd
import csv

def process_fcsv_to_txt(input_file, output_file):
    # Open the input file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Initialize a list to store coordinates
    coordinates = []

    # Parse the lines in the file
    for line in lines:
        # Skip comment lines starting with '#'
        if line.startswith('#'):
            continue

        # Split the line into components
        parts = line.strip().split(',')

        # Ensure the line has enough components for x, y, z
        if len(parts) > 3:
            # Extract x, y, z coordinates
            x, y, z = map(float, parts[1:4])
            coordinates.append((x, y, z))

    # Convert to PyTorch tensor for processing (if needed)
    tensor_coords = torch.tensor(coordinates)

    # Save the coordinates in the specified format
    with open(output_file, 'w') as f:
        for coord in tensor_coords:
            # Format with tab character between coordinates
            f.write(f"{coord[0]:.6f}\t{coord[1]:.6f}\t{coord[2]:.6f}\n")
            
def extract_all_objects_mask(input_nrrd_path: str, output_nrrd_path: str, exclude_label: int = 1):
    """
    Creates a mask of all objects in the NRRD file excluding the specified label (e.g., background).

    Parameters:
        input_nrrd_path (str): Path to the input NRRD file containing segmentation data.
        output_nrrd_path (str): Path to save the output mask NRRD file.
        exclude_label (int): The label to exclude from the mask (e.g., 1 for the first/background segment).

    Returns:
        None
    """
    # Load NRRD data
    data, header = nrrd.read(input_nrrd_path)
    data = np.array(data)  # Ensure the data is a NumPy array

    # Convert data to PyTorch tensor
    tensor_data = torch.tensor(data, dtype=torch.int32)

    # Create a mask for all labels except the excluded label
    mask_tensor = (tensor_data != exclude_label).int()

    # Convert the mask back to a NumPy array
    mask_array = mask_tensor.numpy()

    # Save the mask as a new NRRD file
    nrrd.write(output_nrrd_path, mask_array, header)
    print(f"Mask of all objects (excluding label {exclude_label}) saved to {output_nrrd_path}")

def pad_and_save_images(fixed_image, moving_image, target_shape, fixed_output_path, moving_output_path):
    """
    Pads the fixed and moving images to a specified target shape and saves them as NRRD files.

    Parameters:
    - fixed_image (SimpleITK.Image): The fixed image to pad.
    - moving_image (SimpleITK.Image): The moving image to pad.
    - target_shape (tuple): The target shape (z, y, x).
    - fixed_output_path (str): File path to save the padded fixed image in NRRD format.
    - moving_output_path (str): File path to save the padded moving image in NRRD format.
    """
    def pad_image(image, target_shape):
        """
        Pads a SimpleITK image to the target shape using zero-padding.

        Parameters:
        - image (SimpleITK.Image): The image to pad.
        - target_shape (tuple): The target shape (z, y, x).

        Returns:
        - SimpleITK.Image: The padded image.
        """
        # Get the size of the input image
        input_size = image.GetSize()
        input_spacing = image.GetSpacing()
        input_origin = image.GetOrigin()
        input_direction = image.GetDirection()

        # Calculate padding for each dimension
        pad_width = [
            (0, max(0, target_shape[dim] - input_size[dim])) for dim in range(len(target_shape))
        ]

        # Convert to numpy array for padding
        image_array = sitk.GetArrayFromImage(image)
        padded_array = np.pad(
            image_array,
            pad_width[::-1],  # Reverse for z, y, x to array axis order
            mode='constant',
            constant_values=0
        )

        # Convert back to SimpleITK.Image
        padded_image = sitk.GetImageFromArray(padded_array)

        # Update metadata for new padded image
        padded_image.SetSpacing(input_spacing)
        padded_image.SetOrigin(input_origin)
        padded_image.SetDirection(input_direction)

        return padded_image
    
def add_rows_to_file(file_path, total_number):
    """
    Adds two rows to the beginning of a text file and saves the updated content
    to a new file with `_edited` added to the original file name.

    The first row will be "point", and the second row will be the total_number.

    Args:
        file_path (str): Path to the text file to modify.
        total_number (int): The total number to write as the second row.

    Raises:
        FileNotFoundError: If the input file does not exist.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")

    # Read the file contents
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Prepare the new lines to add
    new_lines = ["point\n", f"{total_number}\n"]

    # Combine the new lines with the original file contents
    updated_lines = new_lines + lines

    # Create the output file path by appending '_edited' before the file extension
    base, ext = os.path.splitext(file_path)
    output_path = f"{base}_edited{ext}"

    # Write the updated content to the new file
    with open(output_path, 'w') as file:
        file.writelines(updated_lines)

    print(f"Updated file saved as: {output_path}")
    
def extract_output_points(input_file_path, output_file_path):
    """
    Extracts the OutputPoint values from the input file and writes them into a new text file.

    Parameters:
        input_file_path (str): Path to the input file containing the data.
        output_file_path (str): Path to the output text file to save the extracted OutputPoints.
    """
    output_points = []

    # Regular expression to match the OutputPoint values
    output_point_pattern = re.compile(r"OutputPoint = \[ ([\d.\-]+) ([\d.\-]+) ([\d.\-]+) \]") #voxel space to get integer result

    # Read the input file and extract OutputPoints
    with open(input_file_path, 'r') as file:
        for line in file:
            match = output_point_pattern.search(line)
            if match:
                output_points.append(
                    f"{float(match.group(1))} {float(match.group(2))} {float(match.group(3))}"
                )

    # Write the extracted OutputPoints to a new text file
    with open(output_file_path, 'w') as txtfile:
        txtfile.write("\n".join(output_points))
        
        
def convert_to_fcsv_with_labels(input_file, output_file, labels_file):
    # Read the input coordinates
    coordinates = []
    with open(input_file, 'r') as file:
        for line in file:
            if line.strip():  # Ignore empty lines
                coordinates.append(list(map(float, line.strip().split())))

    # Convert the coordinates to PyTorch tensors
    coordinates_tensor = coordinates

    # Define header for the .fcsv file
    header = [
        "# Markups fiducial file version = 5.6",
        "# CoordinateSystem = LPS",
        "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID"
    ]

    # Read label information from the provided labels file
    labels = []
    with open(labels_file, 'r') as file:
        for line in file:
            if not line.startswith("#") and line.strip():  # Ignore comments and empty lines
                parts = line.strip().split(',')
                if len(parts) > 11:  # Ensure enough columns exist to fetch the label
                    labels.append(parts[11])  # The label is at index 11

    # Prepare rows for the .fcsv file
    rows = []
    for i, coord in enumerate(coordinates_tensor):
        label = labels[i] if i < len(labels) else i + 1  # Use existing label or default to index+1
        row = [
            1,  # id
            coord[0], coord[1], coord[2],  # x, y, z
            0, 0, 0, 1,  # ow, ox, oy, oz
            1, 1, 1,  # vis, sel, lock
            label,  # label
            "",  # desc
            "2,0"  # associatedNodeID
        ]
        rows.append(row)

    # Write to the output .fcsv file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        for line in header:
            file.write(line + '\n')

        # Write the rows
        writer.writerows(rows)
        
def extract_3d_coordinates(file_path):
    """
    Extracts 3D coordinates from a text file and returns them as a list of tuples.
    
    Parameters:
        file_path (str): Path to the text file containing 3D coordinates.
        
    Returns:
        list: A list of tuples, each containing three floats (x, y, z).
    """
    coordinates = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line into components and convert them to floats
                values = line.split()
                if len(values) == 3:  # Ensure the line has exactly three values
                    x, y, z = map(float, values)
                    coordinates.append((x, y, z))
    except Exception as e:
        print(f"Error reading the file: {e}")
    
    return coordinates

    
def _compute_TRE(targets, actual):
    """
    Computes the Target Registration Error (TRE) between targets and actual arrays in physical coordinates (mm).
    
    Parameters:
        targets (np.ndarray or list): Array or list of target coordinates in mm.
        actual (np.ndarray or list): Array or list of actual coordinates in mm.
        
    Returns:
        float: The mean Target Registration Error in mm.
    """
    # Ensure inputs are converted to NumPy arrays
    targets = np.array(targets)
    actual = np.array(actual)
    
    # Compute the difference vectors and their Euclidean distances
    vectors = targets - actual
    return np.mean(np.sqrt(np.sum(vectors**2, axis=1)))

    
  
def get_landmark(patient_num):
    """Get the landmarks of the patient based on inhale or exhale.

    Args:
        phase (str): Inhale ('i') or exhale ('e').
        patient_num (int): Patient number.

    Returns:
        np.ndarray: The landmarks as a numpy array scaled by the image spacing.
    """


    input_fcsv = f"Training_data_2/cop1/data/NG41{patient_num}_Fiducial_template_ALL.fcsv"
    output_txt = f"Training_data_2/cop1/data/NG41{patient_num}_Fiducial_template_ALL.txt"
    process_fcsv_to_txt(input_fcsv, output_txt)
  
   
    target_array = extract_3d_coordinates(output_txt)
    
    # Scale landmarks by spacing to convert to mm
    return target_array  # Return as numpy array

def register(patient_num):

    input_path = f"Training_data_2/cop1/data/NG41{patient_num}_Segments.seg.nrrd"  # Replace with your input file path
    output_path = f"Training_data_2/cop1/data/NG41{patient_num}_mask.nrrd" # Replace with desired output path
    exclude_label = 0  # Adjust to the label representing the background or first segment
    extract_all_objects_mask(input_path, output_path, exclude_label=exclude_label)
    moved = f"Training_data_2/cop1/data/NG41{patient_num}_RCL5_masked.nrrd"  # Replace with your input file path
    output_dir = fr'par/cop{patient_num}'
    os.makedirs(output_dir, exist_ok=True)
   

    return output_path,moved ,output_dir

def transform(patient_num):
    file_path = f"Training_data_2/cop1/data/NG4108_Fiducial_template_ALL.txt"  # Replace with your input file path
    total_number=98
    add_rows_to_file(file_path,total_number)
    edited_file_path = f"Training_data_2/cop1/data/NG4108_Fiducial_template_ALL_edited.txt"  # Replace with your input file path
    output_dir = fr'par/cop{patient_num}/step2' #<---------------------------------
    transform_params = fr'par/cop{patient_num}/TransformParameters.1.txt'#<---------------------------------
    os.makedirs(output_dir, exist_ok=True)
    return edited_file_path, output_dir,transform_params

def extract (patient_num):
    output_dir = fr'outputpoints'
    os.makedirs(output_dir, exist_ok=True)
    final_path=fr'par/cop{patient_num}/step2/NG41{patient_num}_outputpoints_final.txt'
    extract_output_points(fr'par/cop{patient_num}/step2/outputpoints.txt', final_path)
    output_file = fr'outputpoints/NG41{patient_num}_outputpoints_final.fcsv'  # Replace with your desired output file path
    labels_file = f"Training_data_2/cop1/data/NG41{patient_num}_Fiducial_template_ALL.fcsv"
    convert_to_fcsv_with_labels(final_path, output_file, labels_file)

    input = f"Training_data_2/cop1/data/NG41{patient_num}_Fiducial_template_ALL.txt"
    target_array_fixed = pd.read_csv(input, sep="\t", header=None)
    target_array_fixed = np.asanyarray(target_array_fixed)
    target_array_fixed = np.round(target_array_fixed)[:, :3].astype(int)
    target_array_fixed -= 1 
  
    
    target_array_moved = pd.read_csv(final_path, delim_whitespace=True, header=None, dtype=float).values
    target_array_moved = np.asanyarray(target_array_moved)
    target_array_moved = np.round(target_array_moved)[:, :3].astype(int)
    target_array_moved -= 1 
 
    
    return target_array_fixed, target_array_moved