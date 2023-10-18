import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, GLPNForDepthEstimation
import torch
from PIL import Image
import requests
import numpy as np


IMG_PROCESSOR = AutoImageProcessor.from_pretrained("vinvino02/glpn-kitti")


class GLPNDepthEstimation:
    def __init__(self):
        # we load the GLPN hugging face model: https://huggingface.co/vinvino02/glpn-kitti
        self.torch_model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-kitti")
        self.image_processor = IMG_PROCESSOR

    def infer(self, image):
        # prepare image for the model
        inputs = self.image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.torch_model(**inputs)
            return outputs.predicted_depth

    def export_to_onnx(self, path):
        self.torch_model.eval()

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = self.image_processor(images=image, return_tensors="pt")
        x = inputs.data['pixel_values']
        torch.onnx.export(self.torch_model, x, path)


    def visualize_prediction(self, pred):
        output = np.squeeze(pred.numpy())
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)
        plt.imshow(depth)



