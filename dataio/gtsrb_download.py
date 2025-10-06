import os
import requests
import zipfile

BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
DATA_DIR = "data/gtsrb/"
ZIPS = [
    "GTSRB_Final_Training_Images",
    "GTSRB_Final_Test_Images",
    "GTSRB_Final_Test_GT",
]


def get(name: str) -> None:
    print(f"Downloading {name}...")
    url = BASE_URL + name + ".zip"
    response = requests.get(url)
    with open(DATA_DIR + name + ".zip", "wb") as f:
        f.write(response.content)


def unpack(name: str) -> None:
    print(f"Unpacking {name}...")
    with zipfile.ZipFile(DATA_DIR + name + ".zip", "r") as zip_ref:
        zip_ref.extractall(DATA_DIR + name)


def ensure_data(name: str, force_redownload: bool = False) -> None:
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.isdir(DATA_DIR + name) or force_redownload:
        if not os.path.isfile(DATA_DIR + name + ".zip") or force_redownload:
            get(name)
        unpack(name)


def delete_zip(name: str) -> None:
    if os.path.isfile(DATA_DIR + name + ".zip"):
        os.remove(DATA_DIR + name + ".zip")
    else:
        print(f"{name} not found, skipping deletion.")


def get_data(force_redownload: bool = False, keep_zips: bool = False) -> None:
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    for zip_name in ZIPS:
        ensure_data(zip_name, force_redownload)
        if not keep_zips:
            delete_zip(zip_name)

    print("GTSRB data is ready.")


def clean_data() -> None:
    """
    Creates one directory with all images and a CSV file with labels.
    CSV structure: filename,label
    1. Rename `GT_final_test.csv` to `labels.csv`
    2. Create `images` directory if it doesn't exist
    3. Move all images from `Final_Test/Images` into `images`
    4. Delete `Final_Test` directory
    5. Loop through subdirectories of `Final_Training/Images`:
        a. For each subdirectory, move all images into `images` directory
        b. Append `GT-XXXXX.csv` to `labels.csv`
        c. Delete the subdirectory
    6. Delete `Final_Training` directory
    7. Delete `GTSRB` directory
    """
    # 1. Rename `GT_final_test.csv` to `labels.csv`
    os.rename(
        DATA_DIR + "GTSRB_Final_Test_GT/GT-final_test.csv", DATA_DIR + "labels.csv"
    )
    os.rmdir(DATA_DIR + "GTSRB_Final_Test_GT/")
    # 2. Create `images` directory if it doesn't exist
    if not os.path.isdir(DATA_DIR + "images/"):
        os.makedirs(DATA_DIR + "images/")
    # 3. Move all images from `Final_Test/Images` into `images`
    for img in os.listdir(DATA_DIR + "GTSRB_Final_Test_Images/GTSRB/Final_Test/Images"):
        os.rename(
            DATA_DIR + "GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/" + img,
            DATA_DIR + "images/" + img,
        )
    # 4. Delete `Final_Test` directory
    os.rmdir(DATA_DIR + "GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/")
    os.rmdir(DATA_DIR + "GTSRB_Final_Test_Images/GTSRB/Final_Test/")
    os.remove(DATA_DIR + "GTSRB_Final_Test_Images/GTSRB/Readme-Images-Final-test.txt")
    os.rmdir(DATA_DIR + "GTSRB_Final_Test_Images/GTSRB/")
    os.rmdir(DATA_DIR + "GTSRB_Final_Test_Images/")
    # 5. Loop through subdirectories of `Final_Training/Images`:
    for subdir in os.listdir(
        DATA_DIR + "GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"
    ):
        subdir_path = (
            DATA_DIR
            + "GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/"
            + subdir
            + "/"
        )
        # a. For each subdirectory, move all images into `images` directory
        for img in os.listdir(subdir_path):
            if not img.endswith(".csv"):
                os.rename(
                    subdir_path + img,
                    DATA_DIR + "images/" + subdir + "_" + img,
                )
        # b. Append `GT-XXXXX.csv` to `labels.csv`
        label_file = (
            DATA_DIR
            + f"GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/{subdir}/GT-{subdir}.csv"
        )
        with (
            open(label_file, "r") as lf,
            open(DATA_DIR + "labels.csv", "a") as main_lf,
        ):
            next(lf)  # Skip header
            for line in lf:
                main_lf.write(subdir + "_" + line)
        os.remove(label_file)
        # c. Delete the subdirectory
        os.rmdir(subdir_path)
    # 6. Delete `Final_Training` directory
    os.rmdir(DATA_DIR + "GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/")
    os.rmdir(DATA_DIR + "GTSRB_Final_Training_Images/GTSRB/Final_Training/")
    os.remove(DATA_DIR + "GTSRB_Final_Training_Images/GTSRB/Readme-Images.txt")
    os.rmdir(DATA_DIR + "GTSRB_Final_Training_Images/GTSRB/")
    os.rmdir(DATA_DIR + "GTSRB_Final_Training_Images/")

    print("GTSRB data cleaned and organized.")


if __name__ == "__main__":
    get_data()
    clean_data()
