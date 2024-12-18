import os
import httpx
import shutil

# Create data/raw/mrds directories if they don't exist
data_dir = "data"
raw_dir = os.path.join(data_dir, "raw")
mrds_dir = os.path.join(raw_dir, "mrds")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)
if not os.path.exists(mrds_dir):
    os.makedirs(mrds_dir)

# URL for MRDS data
mrds_url = "https://mrdata.usgs.gov/mrds/mrds-csv.zip"

# Download the zip file
print("Downloading MRDS data...")
response = httpx.get(mrds_url)
zip_path = os.path.join(raw_dir, "mrds.zip")
with open(zip_path, "wb") as f:
    f.write(response.content)

# Extract using shutil (built-in)
print("Extracting zip file...")
shutil.unpack_archive(zip_path, mrds_dir)

# Clean up by removing the zip file
os.remove(zip_path)
print("Download and extraction complete!")
