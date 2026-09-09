import os
import sys
import requests
from concurrent.futures import ThreadPoolExecutor

FILES = [
    "LC09_L2SP_143045_20241106_20241107_02_T1_SR_B4.TIF",
    "LC09_L2SP_143045_20241106_20241107_02_T1_SR_B5.TIF",
    "LC09_L2SP_143045_20241106_20241107_02_T1_SR_B6.TIF",
    "LC09_L2SP_143045_20241106_20241107_02_T1_SR_B7.TIF",
]

BASE_URL = "https://raw.githubusercontent.com/bhavishkulal/AEROGEO-AI/main/raw_data/"
DEST_DIR = "raw_data"

def download_chunk(url, dest_path, start, end, chunk_idx):
    headers = {"Range": f"bytes={start}-{end}"}
    r = requests.get(url, headers=headers, stream=True, timeout=60)
    r.raise_for_status()
    with open(dest_path, "r+b") as f:
        f.seek(start)
        for block in r.iter_content(65536):
            if block:
                f.write(block)
    return chunk_idx

def download_file(filename, num_workers=12):
    os.makedirs(DEST_DIR, exist_ok=True)
    dest_path = os.path.join(DEST_DIR, filename)
    url = BASE_URL + filename
    
    r = requests.head(url, allow_redirects=True, timeout=30)
    r.raise_for_status()
    total_size = int(r.headers.get("content-length", 0))
    print(f"[{filename}] Total size: {total_size / (1024*1024):.2f} MB", flush=True)
    
    if os.path.exists(dest_path) and os.path.getsize(dest_path) == total_size:
        print(f"[{filename}] Already complete!", flush=True)
        return
    
    # Pre-allocate file
    with open(dest_path, "wb") as f:
        f.truncate(total_size)
    
    chunk_size = 4 * 1024 * 1024 # 4 MB chunks
    chunks = []
    start = 0
    idx = 0
    while start < total_size:
        end = min(start + chunk_size - 1, total_size - 1)
        chunks.append((start, end, idx))
        start = end + 1
        idx += 1
        
    print(f"[{filename}] Downloading {len(chunks)} chunks using {num_workers} threads...", flush=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(download_chunk, url, dest_path, c[0], c[1], c[2]) for c in chunks]
        for fut in futures:
            fut.result()
            completed += 1
            if completed % 4 == 0 or completed == len(chunks):
                print(f"[{filename}] {completed}/{len(chunks)} chunks done ({completed*100//len(chunks)}%)", flush=True)
    print(f"[{filename}] Download completed successfully!", flush=True)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        download_file(sys.argv[1])
    else:
        for f in FILES:
            download_file(f)
