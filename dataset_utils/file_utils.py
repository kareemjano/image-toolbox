from pathlib import Path
import zipfile

def unzip_file(zip_path, dest_path):
    print("Unzipping file ", zip_path, "to", dest_path)
    zip_file = zipfile.ZipFile(zip_path, "r")
    zip_file.extractall(path=dest_path)
    zip_file.close()
    # os.remove(zip_path)
    imgs_path = Path(dest_path) / Path(zip_path).stem
    print("Image are in", str(imgs_path))
    return imgs_path