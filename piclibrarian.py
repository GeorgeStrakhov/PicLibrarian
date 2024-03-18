import os
from dotenv import load_dotenv
from pathlib import Path
import csv
from tqdm import tqdm
from PIL import Image
import uuid
import re

from captioner import generate_caption
from llm import humanize, generate_embedding

# Load environment variables
load_dotenv()

def rename_image(file_path):
    file_path = Path(file_path)  # Ensure file_path is a Path object
    directory = file_path.parent
    name, ext = os.path.splitext(file_path.name)
    # Check if name contains only English letters and digits
    if re.match(r'^[a-zA-Z0-9]+$', name):
        new_name = name + ext
    else:
        # If not, use a UUID
        new_name = str(uuid.uuid4()) + ext
    new_file_path = directory / new_name
    file_path.rename(new_file_path)
    return new_file_path

def resize_image(input_path, output_directory, target_width=512):
    """
    Resizes an image to a target width while maintaining aspect ratio and saves it to the specified directory.
    """
    image = Image.open(input_path)
    original_width, original_height = image.size
    aspect_ratio = original_height / original_width
    new_height = int(target_width * aspect_ratio)
    image = image.resize((target_width, new_height), Image.Resampling.LANCZOS)
    output_path = output_directory / input_path.name
    image.save(output_path)

def rename_images(directory):
    """
    Renames all images in the given directory to URL-friendly names.
    """
    for item in directory.iterdir():
        if item.is_file() and item.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            rename_image(item)

def resize_images(input_directory, output_directory, target_width=512):
    """
    Resizes all images in the input directory to the specified width while maintaining the aspect ratio.
    """
    for item in input_directory.iterdir():
        if item.is_file() and item.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            resize_image(item, output_directory, target_width)

def generate_description_and_embedding(image_path):
    caption = generate_caption(image_path)
    description = humanize(caption)
    embedding = generate_embedding(description)
    return description, embedding

def cleanup(directory):
    if os.path.isdir(directory):
        for file in os.listdir(directory):
            os.remove(os.path.join(directory, file))
        os.rmdir(directory)

def main():
    images_directory = Path("./images/")
    resized_directory = images_directory / "resized"
    resized_directory.mkdir(exist_ok=True)

    # Rename and resize images
    rename_images(images_directory)
    resize_images(images_directory, resized_directory)

    # Prepare CSV file
    csv_file = Path("./image_data.csv")
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Caption", "Embedding"])

        for image_path in tqdm(list(resized_directory.iterdir()), desc="Processing images"):
            description, embedding = generate_description_and_embedding(image_path)
            writer.writerow([image_path.name, description, str(embedding)])

    # Cleanup
    cleanup(resized_directory)

if __name__ == "__main__":
    main()

