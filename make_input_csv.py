# Script to set up the input CSV for MedSAM2 inference from a directory of images

import click
from pathlib import Path
import pandas as pd


@click.command()
@click.argument("dataset_name", type=str)
@click.argument("images_path", type=click.Path(exists=True))
@click.option(
    "--image_type",
    type=str,
    default="CT",
    help="Type of images to look for (default: CT)",
)
@click.option(
    "--mask_type",
    type=str,
    default="RTSTRUCT",
    help="Type of masks to look for (default: RTSTRUCT)",
)
def make_medsam2_input_csv(
    dataset_name: str,
    images_path: Path,
    image_type: str = "CT",
    mask_type: str = "RTSTRUCT",
) -> Path:
    """Create a CSV file for MedSAM2 inference from a directory of images.
    Args:
        dataset_name (str): Name of the dataset (used for path for saving CSV).
        images_path (Path): Path to the directory containing image sample directories. Each directory should contain subdirectories for images and masks.
    Returns:
        csv_save_path (Path): Path to the generated CSV file.
    """
    images_path = Path(images_path)
    # Initialize list to hold sample data that will be saved out for medsam2 inference
    samples_to_process = []
    # Loop through each subdirectory in the images_path
    for sample in images_path.iterdir():
        # Check if it's a directory
        if sample.is_dir():
            # Find the image and mask files
            image_path = list(sample.rglob(f"{image_type}*/*.nii.gz"))
            mask_path = list(sample.rglob(f"{mask_type}*/*nii.gz"))

            # if both an image and mask file are present, add this sample to the to be processed list
            if len(image_path) > 0 and len(mask_path) > 0:
                samples_to_process.append(
                    [sample.stem, str(image_path[0]), str(mask_path[0])]
                )

    # Convert the list of samples to a dataframe with the appropriate column names
    samples_df = pd.DataFrame(
        samples_to_process, columns=["ID", "image_path", "mask_path"]
    )
    # Filter out any samples that do not have both an image and a mask path
    suitable_samples_df = samples_df[samples_df["image_path"].map(len) > 0]

    # Setup the output csv path
    image_cohort_name = images_path.stem
    csv_save_path = Path(
        f"data/procdata/{dataset_name}/metadata/{image_cohort_name}_medsam_input.csv"
    )
    # Ensure the parent directories exist
    csv_save_path.parent.mkdir(parents=True, exist_ok=True)
    # Save the dataframe to a CSV file
    suitable_samples_df.to_csv(csv_save_path, index=False)

    return csv_save_path


if __name__ == "__main__":
    make_medsam2_input_csv()