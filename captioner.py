import replicate
from dotenv import load_dotenv
load_dotenv()

def generate_caption(image_path):
    output = replicate.run(
        "lucataco/clip-interrogator:14d81f8a13e8ef87cc9b5eb7d03f5940fc7010e7226e93af612c5f0f4df1a35f",
        input={
            "mode": "fast",
            "image": image_path,
            "clip_model_name": "ViT-bigG-14/laion2b_s39b_b160k"
        }
    )

    return output
