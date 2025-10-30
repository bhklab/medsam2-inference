import os
import pandas as pd
from pathlib import Path

def create_dataset_csv(
    temp_data_path: str = "temp_data",
    output_path: str = "data/temp_dataset_mandible.csv",
    filter_csv: str = "targets.csv",
    filter_column: str = "sample_id",
    roi_name: str = "Mandible_Bone"
):
    """
    Create a dataset CSV file from temp_data directory, filtering by folder suffixes from a CSV column.
    
    Args:
        temp_data_path: Path to the temp_data directory
        output_path: Path where to save the CSV file
        filter_csv: CSV file containing list of IDs to match against folder names
        filter_column: Name of the column in the CSV whose values folder names must end with
    """
    temp_data_path = Path(temp_data_path)

    print(f"Reading filter list from {filter_csv} (column: '{filter_column}')...")
    try:
        filter_df = pd.read_csv(filter_csv)
    except Exception as e:
        print(f"Error reading {filter_csv}: {e}")
        return

    if filter_column not in filter_df.columns:
        print(f"Column '{filter_column}' not found in {filter_csv}")
        return

    patient_ids = set(str(x).strip() for x in filter_df[filter_column].dropna())
    print(len(patient_ids))

    print(f"Scanning {temp_data_path} for matching folders and valid CT/{roi_name} pairs...")

    index_df = pd.read_csv(temp_data_path / f"{temp_data_path.name}_index-simple.csv")

    entries = []
    print(index_df['PatientID'].unique())

    for patient_id in patient_ids:
        # Get CT path from index where modality is CT and PatientID matches
        ct_row = index_df[(index_df['Modality'] == 'CT') & (index_df['PatientID'] == patient_id)]
        if len(ct_row) == 0:
            # Try finding CT file directly from patient ID path pattern
            potential_ct_path = list(temp_data_path.glob(f"{patient_id}*/CT*/CT.nii.gz"))
            if not potential_ct_path:
                print(f"Warning: No CT entry found for patient {patient_id}, skipping...")
                continue
            if len(potential_ct_path) > 1:
                print(f"Warning: Multiple CT files found for patient {patient_id}, using first...")
            ct_path = potential_ct_path[0]
        else:
            ct_path = Path(temp_data_path) / ct_row.iloc[0]['filepath']

        # Get ROI path from index where modality is RTSTRUCT_CT and PatientID matches
        roi_row = index_df[(index_df['matched_rois'] == roi_name) & (index_df['PatientID'] == patient_id)]
        if len(roi_row) == 0:
            # Try finding CT file directly from patient ID path pattern
            potential_roi_paths = list(temp_data_path.glob(f"{patient_id}*/RTSTRUCT*"))
            if not potential_roi_paths:
                print(f"Warning: No RTSTRUCT directory found for patient {patient_id}, skipping...")
                continue
            potential_roi_path = None
            for path in potential_roi_paths:
                test_path = path / f"ROI__[{roi_name}].nii.gz"
                if test_path.exists():
                    potential_roi_path = test_path
                    break
            if potential_roi_path is None:
                print(f"Warning: No valid {roi_name} ROI found in any RTSTRUCT directory for patient {patient_id}, skipping...")
                continue
            if not potential_roi_path:
                print(f"Warning: No {roi_name} entry found for patient {patient_id}, skipping...")
                continue
            roi_path = potential_roi_path
        else:
            roi_path = Path(temp_data_path) / roi_row.iloc[0]['filepath']
        
        entries.append({
            "ID": patient_id,
            "image_path": str(ct_path),
            "mask_path": str(roi_path)
        })

    df = pd.DataFrame(entries)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved dataset CSV to: {output_path}")
    print(f"\n🔍 Preview:")
    print(df.head())


if __name__ == "__main__":
    # roi_names = ['Brainstem', 'SpinalCord', 'Esophagus', 'Larynx', 'Mandible_Bone', 'Parotid_L', 'Parotid_R', 'Cochlea_L', 'Cochlea_R',
    #         'BrachialPlex_R', 'BrachialPlex_L', 'Lens_L', 'Lens_R', 'Eye_L', 'Eye_R', 'Nrv_Optic_L', 'Nrv_Optic_R', 'OpticChiasm', 'Lips']
            
    # for roi_name in roi_names:
    #     create_dataset_csv(
    #         temp_data_path="/cluster/projects/radiomics/PublicDatasets/procdata/HeadNeck/TCIA_RADCURE/images/mit_RADCURE",
    #         output_path=f"data/radcure_dataset_{roi_name.lower()}.csv",
    #         filter_csv="radcure_nnunet_with_clinical.csv",
    #         filter_column="patient",
    #         roi_name=roi_name
    #     )

    create_dataset_csv(
        temp_data_path="/cluster/projects/radiomics/PublicDatasets/procdata/HeadNeck/TCIA_RADCURE/images/mit_RADCURE",
        output_path=f"data/radcure_dataset_gtvp.csv",
        filter_csv="radcure_nnunet_with_clinical.csv",
        filter_column="patient",
        roi_name="GTVp"
    )
