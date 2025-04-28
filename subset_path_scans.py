import os
import shutil
import pandas as pd

# Paths
training_scans = "L:/Basic/divi/jstoker/slicer_pdac/Master Students WS 24/Martijn/data/Training/paired_scans"
path_training_scans = "L:/Basic/divi/jstoker/slicer_pdac/Master Students WS 24/Martijn/data/Training/paired_scans_path_resp"
training_labels_path = "L:/Basic/divi/jstoker/slicer_pdac/Master Students WS 24/Martijn/data/Training/training_labels_path.csv"


# Read the CSV file to get the list of patient IDs
df = pd.read_csv(training_labels_path)
patient_ids = df['SubjectKey'].tolist()

# Copy scans for the listed patient IDs
for scan in os.listdir(training_scans):
    patient_id = int(scan.split("_")[0][-3:])
    if patient_id in patient_ids:
        # Copy the scan to the destination directory
        src_path = os.path.join(training_scans, scan)
        shutil.copy(src_path, path_training_scans)
    