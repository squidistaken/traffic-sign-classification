import os
import requests
import zipfile
import logging

logger = logging.getLogger("GTSRBDownloader")

logger.setLevel(logging.INFO)

formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
console_handler = logging.StreamHandler()

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

log_dir = "../logs"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

file_handler = logging.FileHandler(os.path.join(log_dir, "gtsrb_download.log"))

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

BASE_URL = (
    "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/")
DATA_DIR = "../data/gtsrb/"
ZIPS = [
    "GTSRB_Final_Training_Images",
    "GTSRB_Final_Test_Images",
    "GTSRB_Final_Test_GT",
]


def get(name: str) -> None:
    """Download a file from the base URL.

    Args:
        name (str): The name of the file to download.
    """
    zip_path = DATA_DIR + name + ".zip"
    logger.info(f"Downloading {name}...")
    url = BASE_URL + name + ".zip"
    response = requests.get(url)

    with open(zip_path, "wb") as f:
        f.write(response.content)

    logger.info(f"{name} downloaded successfully.")


def unpack(name: str) -> None:
    """Unpack a zip file.

    Args:
        name (str): The name of the zip file to unpack.
    """
    target_dir = DATA_DIR + name
    zip_path = DATA_DIR + name + ".zip"
    logger.info(f"Unpacking {name}...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    logger.info(f"{name} unpacked successfully.")


def get_data(keep_zips: bool = False) -> None:
    """Download and unpack all dataset files.

    Args:
        keep_zips (bool, optional): Whether to keep the zip files after
                                    unpacking. Defaults to False.
    """
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    for zip_name in ZIPS:
        get(zip_name)
        unpack(zip_name)

        if not keep_zips:
            delete_zip(zip_name)

    logger.info("GTSRB data is ready.")


def delete_zip(name: str) -> None:
    """Delete a zip file if it exists.

    Args:
        name (str): The name of the zip file to delete.
    """
    zip_path = DATA_DIR + name + ".zip"

    if os.path.isfile(zip_path):
        os.remove(zip_path)
        logger.info(f"Deleted {zip_path}.")
    else:
        logger.info(f"{zip_path} not found, skipping deletion.")


def clean_data() -> None:
    """Clean and organize the downloaded dataset files."""
    if not os.path.exists(DATA_DIR + "GTSRB_Final_Test_GT/GT-final_test.csv"):
        logger.info(
            "Data already cleaned or missing expected structure, skipping.")

        return

    os.rename(
        DATA_DIR + "GTSRB_Final_Test_GT/GT-final_test.csv",
        DATA_DIR + "labels.csv"
    )
    os.rmdir(DATA_DIR + "GTSRB_Final_Test_GT/")

    if not os.path.isdir(DATA_DIR + "images/"):
        os.makedirs(DATA_DIR + "images/")

    for img in os.listdir(DATA_DIR
                          + "GTSRB_Final_Test_Images/GTSRB/Final_Test/Images"):
        os.rename(
            DATA_DIR + "GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/"
            + img,
            DATA_DIR + "images/" + img,
        )

    os.rmdir(DATA_DIR + "GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/")
    os.rmdir(DATA_DIR + "GTSRB_Final_Test_Images/GTSRB/Final_Test/")
    os.remove(DATA_DIR +
              "GTSRB_Final_Test_Images/GTSRB/Readme-Images-Final-test.txt")
    os.rmdir(DATA_DIR + "GTSRB_Final_Test_Images/GTSRB/")
    os.rmdir(DATA_DIR + "GTSRB_Final_Test_Images/")

    for subdir in os.listdir(
        DATA_DIR + "GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"
    ):
        subdir_path = (
            DATA_DIR
            + "GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/"
            + subdir
            + "/"
        )

        for img in os.listdir(subdir_path):
            if not img.endswith(".csv"):
                os.rename(
                    subdir_path + img,
                    DATA_DIR + "images/" + subdir + "_" + img,
                )

        label_file = (
            DATA_DIR
            + "GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/"
            + f"{subdir}/GT-{subdir}.csv"
        )

        with (
            open(label_file, "r") as lf,
            open(DATA_DIR + "labels.csv", "a") as main_lf,
        ):
            next(lf)
            for line in lf:
                main_lf.write(subdir + "_" + line)

        os.remove(label_file)
        os.rmdir(subdir_path)

    os.rmdir(DATA_DIR
             + "GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/")
    os.rmdir(DATA_DIR
             + "GTSRB_Final_Training_Images/GTSRB/Final_Training/")
    os.remove(DATA_DIR
              + "GTSRB_Final_Training_Images/GTSRB/Readme-Images.txt")
    os.rmdir(DATA_DIR
             + "GTSRB_Final_Training_Images/GTSRB/")
    os.rmdir(DATA_DIR
             + "GTSRB_Final_Training_Images/")
    logger.info("GTSRB data cleaned and organized.")


if __name__ == "__main__":
    get_data()
    clean_data()
