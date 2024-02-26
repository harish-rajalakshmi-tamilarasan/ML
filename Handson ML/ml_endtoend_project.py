from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path(r"E://datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path=r"E://datasets")
    return pd.read_csv(Path(r"E://datasets/housing/housing.csv"))

df = pd.read_csv(Path(r"E://datasets/housing/housing.csv"))

print(df["ocean_proximity"].value_counts())