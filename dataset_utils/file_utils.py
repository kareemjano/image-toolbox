from pathlib import Path
import zipfile
import os
import requests
import urllib
from tqdm import tqdm

def unzip_file(zip_path, dest_path):
    print("Unzipping file ", zip_path, "to", dest_path)
    zip_file = zipfile.ZipFile(zip_path, "r")
    zip_file.extractall(path=dest_path)
    zip_file.close()
    imgs_path = Path(dest_path) / Path(zip_path).stem
    print("Image are in", str(imgs_path))
    return imgs_path

def download_file_from_google_drive(destination, full_url=None, id=None):
    if (full_url is not None and 'google' in full_url) or id is not None:
        URL = "https://docs.google.com/uc?export=download"
        id = full_url.split('id=')[-1] if id is None else id
        print('id', id)
        session = requests.Session()

        response = session.get(URL, params = { 'id' : id }, stream = True)
        token = get_confirm_token(response)

        while token is not None:
            params = { 'id' : id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)
            token = get_confirm_token(response)

        save_response_content(response, destination)
    else:
        r = requests.get(full_url, allow_redirects=True)
        with open(destination, 'wb') as f:
            f.write(r.content)

def get_confirm_token(response):
    print(response.cookies.items())
    for key, value in response.cookies.items():
        print("print key", key)
        if key.startswith('download_warning'):
            return value

    print('Downloading is done!')
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=destination) as t:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    t.update(CHUNK_SIZE)

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_zip(url, zip_filename, target_dir):
    zip_path = os.path.join(target_dir, zip_filename)
    if zip_filename not in os.listdir(target_dir):
        print('\ndownloading zip file...')

        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, zip_path, reporthook=t.update_to)
    else:
        print('Dir is not empty')

    return zip_path