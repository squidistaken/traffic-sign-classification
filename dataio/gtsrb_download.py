import os
import requests
import zipfile

BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
DATA_DIR = "data/gtsrb/"


def get(name: str) -> None:
    print(f"Downloading {name}...")
    url = BASE_URL + name
    response = requests.get(url)
    with open(DATA_DIR + name, "wb") as f:
        f.write(response.content)


def unpack(name: str) -> None:
    print(f"Unpacking {name}...")
    with zipfile.ZipFile(DATA_DIR + name, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)


def ensure_data(name: str, force_redownload: bool = False) -> None:
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.isfile(DATA_DIR + name) or force_redownload:
        get(name)
    unpack(name)


def get_data(force_redownload: bool = False) -> None:
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    ensure_data("GTSRB_Final_Training_Images.zip", force_redownload)
    ensure_data("GTSRB_Final_Test_Images.zip", force_redownload)
    ensure_data("GTSRB_Final_Test_GT.zip", force_redownload)

    print("GTSRB data is ready.")


if __name__ == "__main__":
    get_data(force_redownload=False)
