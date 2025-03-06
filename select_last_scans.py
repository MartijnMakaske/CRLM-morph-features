import os
import shutil

def filter_last_files(source_dir, dest_dir):
    # Get the list of files and sort them alphabetically
    file_list = os.listdir(source_dir)
    file_list.sort()

    last_patient_id = file_list[0].split("_")[0]
    last_filename = None

    for filename in file_list:
        patient_id = filename.split("_")[0]
        if patient_id != last_patient_id:
            # Copy the last file of the previous patient
            src_file_path = os.path.join(source_dir, last_filename)
            dest_file_path = os.path.join(dest_dir, last_filename)
            shutil.copy(src_file_path, dest_file_path)
            print(f"Copied {last_filename} to {dest_dir}")

        last_patient_id = patient_id
        last_filename = filename

    # Copy the last file of the last patient
    src_file_path = os.path.join(source_dir, last_filename)
    dest_file_path = os.path.join(dest_dir, last_filename)
    shutil.copy(src_file_path, dest_file_path)
    print(f"Copied {last_filename} to {dest_dir}")

def sanity_check():
    # Check if each scan has a corresponding segmentation
    scan_dir = "L:/Basic/divi/jstoker/slicer_pdac/Master Students WS 24/Martijn/data/last scan niftis cropped"
    seg_dir = "L:/Basic/divi/jstoker/slicer_pdac/Master Students WS 24/Martijn/data/last segmentations cropped"

    scan_files = os.listdir(scan_dir)
    seg_files = os.listdir(seg_dir)

    for scan_file in scan_files:
        splitted_scan = scan_file.split("_")
        scan_filename = splitted_scan[0] + "_" + splitted_scan[1]
        seg_file = f"{scan_filename}.nii.gz"
        if seg_file not in seg_files:
            print(f"Segmentation file {seg_file} not found for scan {scan_file}")



#select last scan of each patient
source_dir = "L:/Basic/divi/jstoker/slicer_pdac/Master Students WS 24/Martijn/data/scan niftis cropped"
dest_dir = "L:/Basic/divi/jstoker/slicer_pdac/Master Students WS 24/Martijn/data/last scan niftis cropped"

filter_last_files(source_dir, dest_dir)

#select last segmentation of each patient
source_dir = "L:/Basic/divi/jstoker/slicer_pdac/Master Students WS 24/Martijn/data/labels made by AI model cropped"
dest_dir = "L:/Basic/divi/jstoker/slicer_pdac/Master Students WS 24/Martijn/data/last segmentations cropped"

filter_last_files(source_dir, dest_dir)

#perform a sanity check 
#checks whether each scan has a corresponding segmentation
sanity_check()

#output of sanity check: (probably exlude)
#Segmentation file CAESAR269_2.nii.gz not found for scan CAESAR269_2_0000.nii.gz
#Segmentation file CAESAR581_1.nii.gz not found for scan CAESAR581_1_0000.nii.gz

