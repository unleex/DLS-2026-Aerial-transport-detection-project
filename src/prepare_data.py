import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "pkdarabi/bone-fracture-detection-computer-vision-project", output_dir="dataset"
)

print("Path to dataset files:", path)
