import streamlit as st

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms

from model import ResNet9

# GD_PATH = "https://drive.google.com/file/d/16W9ae5-QQ_P6EHDiUO096bWWp5-3bG6x"
MODEL_PATH = "dump/tomato-disease-model.pth"

CLASSES = ['Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']


def load_model():
	model = ResNet9(3, len(CLASSES))
	model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
	model.eval()
	return model


def predict_image(img, model):
    xb = img.unsqueeze(0) # Convert to a batch of 1
    yb = model(xb) # Do model inference
    _, preds  = torch.max(yb, dim=1) # Pick index with highest probability
    return CLASSES[preds[0].item()]


model = load_model()

st.title("Tomato Disease Classifier")
file = st.file_uploader("Upload an image of a tomato leaf", type=["jpg", "png"])

if file is None:
	st.text("Waiting for upload....")

else:
	slot = st.empty()
	slot.text("Running inference....")

	test_image = Image.open(file)

	st.image(test_image, caption="Input Image", width = 400)

	result = predict_image(transforms.ToTensor()(test_image), model)
	if result == CLASSES[9]:
		output = "This tomato leaf is healthy."
	elif result == CLASSES[0]:
		output = "This tomato leaf is affected by the Bacterial Spot disease !\n\n----------\n\nManagement\n\n----------\n\nResistant varieties: Planting resistant varieties can help reduce the impact.\n\nCultural practices: Proper spacing, crop rotation, and avoiding overhead irrigation can minimize the spread.\n\nPhysical controls: Removing infected plants and debris.\n\nPesticides: Using appropriate pesticides as a last resort."
	elif result == CLASSES[1]:
		output = "This tomato leaf is affected by the Early Blight disease !\n\n----------\n\nManagement\n\n----------\n\nResistant varieties: Planting resistant cultivars can help reduce disease impact.\n\nCultural practices: Crop rotation, proper spacing, and avoiding overhead irrigation can minimize spread.\n\nSanitation: Removing and destroying infected plant debris.\n\nFungicides: Applying fungicides as a preventive measure, especially in wet conditions."
	elif result == CLASSES[2]:
		output = "This tomato leaf is affected by the Late Blight disease !\n\n----------\n\nManagement\n\n----------\n\nResistant varieties: Planting resistant cultivars can help reduce the impact.\n\nCultural practices: Crop rotation, proper spacing, and avoiding overhead irrigation can minimize spread.\n\nSanitation: Removing and destroying infected plant debris.\n\nFungicides: Applying fungicides as a preventive measure, especially in wet conditions."
	elif result == CLASSES[3]:
		output = "This tomato leaf is affected by the Leaf Mold disease !\n\n----------\n\nManagement\n\n----------\n\nResistant varieties: Use certified disease-free or treated seeds.\n\nCultural practices: Maintain proper ventilation, avoid overhead watering, and sanitize greenhouses between crop seasons.\n\nSanitation: Remove and destroy all crop debris post-harvest.\n\nFungicides: Apply fungicides at the first sign of infection, following the manufacturer’s instructions."
	elif result == CLASSES[4]:
		output = "This tomato leaf is affected by the Septoria Leaf Spot disease !\n\n----------\n\nManagement\n\n----------\n\nCultural practices: Rotate crops, avoid overhead watering, and ensure good air circulation around plants.\n\nSanitation: Remove and destroy infected plant debris.\n\nFungicides: Apply fungicides as a preventive measure, especially in wet conditions."
	elif result == CLASSES[5]:
		output = "This tomato leaf is affected by the Two-Spotted Spider Mites disease !\n\n----------\n\nManagement\n\n----------\n\nMonitoring: Regularly scout for mites, especially during hot, dry weather. Use a hand lens to check the undersides of leaves.\n\nCultural practices: Maintain proper irrigation to reduce dust, which can exacerbate mite problems. Remove and destroy infested plant debris.\n\nBiological control: Introduce natural predators like predatory mites.\n\nChemical control: Use miticides as a last resort, following the manufacturer’s instructions."
	elif result == CLASSES[6]:
		output = "This tomato leaf is affected by the Target Spot disease !\n\n----------\n\nManagement\n\n----------\n\nResistant varieties: Planting resistant cultivars can help reduce the impact.\n\nCultural practices: Crop rotation, proper spacing, and avoiding overhead irrigation can minimize spread.\n\nSanitation: Removing and destroying infected plant debris.\n\nFungicides: Applying fungicides as a preventive measure, especially in humid conditions."
	elif result == CLASSES[7]:
		output = "This tomato leaf is affected by the Yellow Leaf Curl Virus !\n\n----------\n\nManagement\n\n----------\n\nResistant varieties: Planting resistant cultivars can help mitigate the impact.\n\nCultural practices: Implement crop rotation, proper spacing, and avoid overhead irrigation to reduce spread.\n\nSanitation: Remove and destroy infected plant debris.\n\nMonitoring and control: Regularly scout for whiteflies and use insecticides if necessary to control their population."
	elif result == CLASSES[8]:
		output = "This tomato leaf is affected by the Mosaic Virus !\n\n----------\n\nManagement\n\n----------\n\nResistant varieties: Planting resistant cultivars is one of the most effective ways to manage ToMV.\n\nSanitation: Regularly clean and disinfect tools, and remove and destroy infected plant debris.\n\nCultural practices: Use certified disease-free seeds and practice crop rotation to reduce the risk of infection."
	else:
		output = "Something went wrong..."

	slot.text("Done")
	st.success(output)

