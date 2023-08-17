from loadmodels import vit_processor,vit_model 
from PIL import Image 
import requests
from main import bytes_data
# url = "https://images.pexels.com/photos/1170986/pexels-photo-1170986.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
image = bytes_data
text = "Describe the image"


# processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
# model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare inputs
encoding = vit_processor(image, text, return_tensors="pt")

# forward pass
outputs = vit_model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
answer = vit_model.config.id2label[idx]
print("Predicted answer:", answer)