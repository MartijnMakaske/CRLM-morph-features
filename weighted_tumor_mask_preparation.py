# Imports
import numpy as np 
import nibabel as nib
import os
import shutil

from scipy.ndimage import binary_dilation
from scipy.ndimage import generate_binary_structure

# Define all usefull paths 
all_scans_path = "/scratch/bmep/mfmakaske/all_scans"
all_segmentations_path = "/scratch/bmep/mfmakaske/all_segmentations"

weigthed_scans_path = "/scratch/bmep/mfmakaske/weighted_scans"
paired_weighted_scans_path = "/scratch/bmep/mfmakaske/paired_weighted_scans"

def create_weighted_tumor_mask(scans_path, segmentations_path, output_path):
    """
    Segments the liver, creates a bounding box based on the liver, and segments only the tumors within the bounding box.
    """
    # Generates a 3D spherical-like connectivity structure
    structure = generate_binary_structure(3, 1)  # 3D, with connectivity=1

    for scan in os.listdir(scans_path):
        print(f"Currently processing: {scan}")

        # Load image and corresponding segmentation
        image = nib.load(os.path.join(scans_path, scan))

        segm_filename = scan.replace("_0000", "")
        segmentation = nib.load(os.path.join(segmentations_path, segm_filename))

        image_data = image.get_fdata()
        segmentation_data = segmentation.get_fdata()

        # Create a liver mask (labels 12 and 13)
        liver_mask = (segmentation_data == 12) | (segmentation_data == 13)

        # Apply the liver mask to the image
        liver_image = np.copy(image_data)
        liver_image[~liver_mask] = -1000

        # Find the indices of the liver mask
        mask_indices = np.argwhere(liver_mask)

        # Calculate the bounding box for the liver
        min_indices = mask_indices.min(axis=0)
        max_indices = mask_indices.max(axis=0)

        # Crop the liver image using the bounding box
        cropped_liver_image = liver_image[min_indices[0]:max_indices[0]+1, min_indices[1]:max_indices[1]+1, min_indices[2]:max_indices[2]+1]

        # Crop the segmentation data using the same bounding box
        cropped_segmentation_data = segmentation_data[min_indices[0]:max_indices[0]+1, min_indices[1]:max_indices[1]+1, min_indices[2]:max_indices[2]+1]

        # Create a tumor mask (label 13) within the cropped liver region
        tumor_mask = (cropped_segmentation_data == 13)

        # Apply binary dilation to the tumor mask
        dilated_tumor_mask = binary_dilation(tumor_mask, structure=structure, iterations=10)

        # Apply the dilated tumor mask to the cropped liver image
        # CHANGE THIS
        tumor_image = np.copy(cropped_liver_image)
        tumor_image[~dilated_tumor_mask.astype(bool)] = -1000

        # Create a new NIfTI image for the tumor segmentation
        new_image = nib.Nifti1Image(tumor_image, affine=image.affine, header=image.header)

        # Save the new NIfTI image to a file with the original name
        new_filename = (scan.split(".")[0])[0:-5] + "_tumor.nii.gz"
        output_file_path = os.path.join(output_path, new_filename)
        nib.save(new_image, output_file_path)


def create_weighted_tumor_mask_pairs(weighted_scans_path, output_path):
    """
    copies scans of patients that have both 0 and 1 scans
    """

    # Create a dictionary to track patient scans
    patient_scans = {}

    # Iterate through all scans
    for scan in os.listdir(weighted_scans_path):
        # Extract the patient ID and scan type (0 or 1)

        patient_id, scan_type = scan.split("_")[0], scan[0:-7].split("_")[1]

        # Add the scan type to the patient's record
        if patient_id not in patient_scans:
            patient_scans[patient_id] = set()
        patient_scans[patient_id].add(scan_type)

    # Move scans of patients that have both 0 and 1 scans
    for patient_id, scan_types in patient_scans.items():
        if "0" in scan_types and "1" in scan_types:
            first_scan = f"{patient_id}_0_tumor.nii.gz"
            second_scan = f"{patient_id}_1_tumor.nii.gz"
            shutil.copy(src=os.path.join(weighted_scans_path, first_scan), dst=os.path.join(output_path, first_scan))
            shutil.copy(src=os.path.join(weighted_scans_path, second_scan), dst=os.path.join(output_path, second_scan))


create_weighted_tumor_mask(all_scans_path, all_segmentations_path, weigthed_scans_path)
create_weighted_tumor_mask_pairs(weigthed_scans_path, paired_weighted_scans_path)
