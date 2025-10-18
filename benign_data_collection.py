import random
import gzip
import io
import requests
from tqdm import tqdm 
import pyarrow.parquet as pq
# import pandas as pd
import tempfile
from multiprocessing import Pool, cpu_count
from functools import partial
import sys

CRAWL_ID = ["CC-MAIN-2025-30", "CC-MAIN-2025-26", "CC-MAIN-2025-21", "CC-MAIN-2025-18", "CC-MAIN-2025-13", "CC-MAIN-2025-05", "CC-MAIN-2024-51", "CC-MAIN-2024-46", "CC-MAIN-2024-42"]
SAMPLE_SIZE = 1000     
OUTPUT_FILE = "urls.txt"       
MAX_PROCESSES = 4

def get_index_paths(crawl_id):
    index_url = f"https://data.commoncrawl.org/crawl-data/{crawl_id}/cc-index-table.paths.gz"
    response = requests.get(index_url, stream=True)
    response.raise_for_status()
    
    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
        return [line.decode().strip() for line in gz]

def save_urls(urls, filename):
    with open(filename, "w") as f:
        f.write("\n".join(urls))
    print(f"Saved {len(urls):,} URLs to {filename}")


except_count = 0

def process_single_file(path, sample_prob):
    urls = []
    parquet_url = f"https://data.commoncrawl.org/{path}"
    
    global except_count
    if except_count > 5:
        print('Too many exception')
        sys.exit(0)
    
    try:
        response = requests.get(parquet_url, stream=True, timeout=30)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile() as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp.flush()
            
            parquet_file = pq.ParquetFile(tmp.name)
            for i in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(i, columns=["url", "fetch_status", "content_mime_type"])
                df = table.to_pandas()
                
                # Filter and sample valid HTML URLs
                valid_rows = df[#(df['fetch_status'] == 200) & 
                               (df['content_mime_type'] == 'text/html')]
                
                for url in valid_rows['url']:
                    if random.random() <= sample_prob:
                        urls.append(url)
                        
    except Exception as e:
        print(f"Skipping {path} due to error: {str(e)}")
        except_count += 1
    
    return urls

def sample_urls_parallel(paths, sample_size):
    sample_prob = 0.0001
    
    # Process files in parallel
    sampled = []
    process_func = partial(process_single_file, sample_prob=sample_prob)
    
    with Pool(processes=min(MAX_PROCESSES, len(paths)), maxtasksperchild=5) as pool:
        results = pool.imap_unordered(process_func, paths)
        
        for result in tqdm(results, total=len(paths), desc="Processing files"):
            sampled.extend(result)
    
    return sampled

if __name__ == "__main__":
    
    i = 0
    while i < len(CRAWL_ID):
        paths = get_index_paths(CRAWL_ID[i])
        print(f"Found {len(paths):,} index partitions")
        
        # urls = sample_urls(paths, SAMPLE_SIZE)
        urls = sample_urls_parallel(paths, SAMPLE_SIZE)
        out = '' + CRAWL_ID[i] + '.txt'
        save_urls(urls, out)
        i = i+1