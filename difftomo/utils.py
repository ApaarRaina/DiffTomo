import os
from pathlib import Path
from urllib.parse import urlparse
import zipfile
import requests
from scipy.io import loadmat
from tqdm import tqdm
import h5py
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def load_mat(path):
    path = Path(path)

    try:
        return loadmat(path)

    except NotImplementedError:
        with h5py.File(path, "r") as f:
            return {
                key: np.array(f[key])
                for key in f.keys()
                if not key.startswith("#")
            }

def download_chunk(byte_range):
    start, end = byte_range

    headers = {
        "Range": f"bytes={start}-{end}"
    }

    with requests.get(
        url,
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=timeout,
    ) as response:

        if response.status_code != 206:
            raise RuntimeError(
                f"Expected HTTP 206 for range "
                f"{start}-{end}, got "
                f"{response.status_code}"
            )

        with destination.open("r+b") as f:
            f.seek(start)

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    f.write(chunk)

def download_file(
    url,
    destination,
    workers=8,
    chunk_size=4 * 1024 * 1024,
    timeout=60,
):
    destination = Path(destination)

    if destination.exists():
        return 1

    destination.parent.mkdir(parents=True, exist_ok=True)

    # Get the total file size
    response = requests.head(
        url,
        allow_redirects=True,
        timeout=timeout,
    )

    response.raise_for_status()

    content_length = response.headers.get("Content-Length")

    if content_length is None:
        # Fall back to normal streaming download
        with requests.get(
                url,
                stream=True,
                allow_redirects=True,
                timeout=timeout,
        ) as response:
            response.raise_for_status()

            with destination.open("wb") as f:
                for chunk in tqdm(
                        response.iter_content(chunk_size=chunk_size),
                        desc=destination.name,
                        unit="B",
                        unit_scale=True,
                ):
                    if chunk:
                        f.write(chunk)

        return 1

    total_size = int(content_length)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(
            tqdm(
                executor.map(download_chunk, ranges),
                total=len(ranges),
                desc=destination.name,
                unit="chunk",
            )
        )

    return 1

def extract_zip(path, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(path) as zip_ref:
        zip_ref.extractall(destination)
        return 1

    return 0

