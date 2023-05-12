from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel
import io
import json
import requests
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import openai



model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(images_list,limit):
  images = []
  for image in images_list:
    i_image = image
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  
  	# Caption generation
  print(preds[0])

  openai.api_key = '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'
  openai_model = "text-davinci-002"

  caption_prompt = ("Generate "+str(limit)+ "small, interesting and unique captions for social media for the description that is "+preds[0])
  
  response = openai.Completion.create(
  engine = openai_model,
  prompt = caption_prompt,
  max_tokens = limit*20,
  n = 1,
  stop = None,
  temperature = 0,
  top_p = 1
  )

  caption = response.choices[0].text.strip().split("\n")

  return(caption)


app = FastAPI(title="Generate Image Caption", description="Upload your image and get an interesting caption for your image.")

class ImageCaption(BaseModel):
    caption: str

@app.get("/")
def index():
    return ("Welcome")

@app.post("/predict/", response_model=ImageCaption)
def predict(file: UploadFile = File(...),limit=3):
    # Load the image file into memory
    contents = file.file.read()
    image = Image.open(io.BytesIO(contents))
    result = predict_step([image],int(limit))
    return JSONResponse(content={"caption": result})

