import json
import logging
import torch
import numpy as np
from sagemaker_inference import encoder
from diffusers import StableDiffusionPipeline

IMAGEARRAY = "image_array"

class TextToImageModel :

    def __init__(self) -> None :
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logging.info(f"Using device: {self.device}")
        model_name = "CompVis/stable-diffusion-v1-4"
        self.model = StableDiffusionPipeline.from_pretrained(model_name,torch_dtype=torch.float16)
        self.model = self.model.to(self.device)
        logging.info(f"Commander I am Here")

    def __call__(self, text :str) -> str :
        with torch.inference_mode():
            image = self.model(text).images[0]
            image_array = np.array(image).tolist()

        return image_array


def model_fn(model_dir: str) -> TextToImageModel:
    try:
        return TextToImageModel()
    except Exception :
        logging.exception(f"Failed to load model from: {model_dir}")
        raise

def transform_fn(sentiment_analysis: TextToImageModel, input_data: bytes, content_type: str, accept: str) -> bytes:
    payload = json.loads(input_data)
    text = payload["text"]
    model_output = sentiment_analysis(text=text)
    output = {IMAGEARRAY : model_output}
    return encoder.encode(output,accept)
            


