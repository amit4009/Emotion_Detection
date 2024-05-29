
import gdown

# Replace with your file ID from Google Drive
file_id = '18mR9u6-MGDuH7PMoNL8TPB1uiOuis9QG'
# Desired output file name
output = 'ResNet50_Transfer_Learning.keras'

# Download the file
try:
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=False)
    print("File downloaded successfully!")
except Exception as e:
    print(f"Error downloading file: {e}")

