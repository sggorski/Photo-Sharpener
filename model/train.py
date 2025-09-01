from dataset import *
from load import *
from build import *
from loss import ssim_loss
from  datetime import datetime
import os

def get_model_name(prefix="model"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.keras"

def train(path,epochs=10, steps=10):
    paths = get_image_paths(path)
    ds = get_dataset(paths)
    model = build_model("vgg19")
    model.compile(
        optimizer="nadam",
        loss="mae"
    )
    print("Started training...")
    model.fit(ds, epochs=epochs, steps_per_epoch=steps)
    print("Finished training...")
    path = os.path.join("trained", get_model_name())
    model.save(path)
    print("Model saved:" + path)

if __name__ == '__main__':
    PATH = r"C:\Users\sggor\Personal\photo_sharpener\images\test2017\*.jpg"
    train(PATH,1,1)