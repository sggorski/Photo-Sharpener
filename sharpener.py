import sys
import os
from model.inference import inference_one
from PIL import Image
import matplotlib.pyplot as plt

def main():
    """
        Performs inference on a single image and saves it in a proper path.
        Validates input data.
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    in_img = sys.argv[1]
    out_img = sys.argv[2]
    if not os.path.exists(in_img):
        print("Can't find input image")
    if not os.path.isdir(os.path.dirname(out_img)):
        print("Can't find output destination")
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"}
    _, ext = os.path.splitext(out_img.lower())
    if ext not in valid_ext:
        print("Invalid extension for the output image. Accepted: ", valid_ext)
    else:
        image = Image.open(in_img).convert("RGB")
        output = inference_one(image, "model/trained/model_latest.keras")
        plt.imsave(out_img, output)
        print("Photo saved at: ", out_img)

if __name__ == "__main__":
    main()
