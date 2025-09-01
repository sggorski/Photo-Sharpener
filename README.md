# Photo Sharpener Project

## Description
This project focuses on creating a model that enhances image quality by removing blur and other imperfections.  

The development process included several stages:  
- **Small initial model** – simple CNN to test basic image sharpening.  
- **U-Net-like model** – encoder-decoder with skip connections to capture local and global features.  
- **ResUNet model** – added residual blocks for more efficient learning of corrections.  
- **Improved data preparation** – original model learned only one type of blur, so randomness was added in preprocessing to generalize better.  
- **Transfer learning with VGG19** – final model based on VGG19 architecture provided the best results.  

Below you can see how my model performs:

![Model Example](examples/examples2.png)  

>  Works best on images around **480x584 px** (training resolution). The model is not perfect and may sometimes misinterpret colors, especially green. Given limited resources, this is the best achievable performance for me. More advanced models (like Google’s) can yield better results. The main reason of this project was to learn something rather than create 100% working solution.

---

## How to Run

1. Download and extract the project ZIP.  
2. Create a new Python virtual environment.  
3. Install dependencies:  
```bash
pip install -r requirements.txt
```
4. Run the sharpener script:
```bash
python sharpener.py <path_to_input_image> <output_image_name>
```

## Tech Stack

- **TensorFlow** – neural network building and training  
- **OpenCV** – image I/O and preprocessing  
- **Jupyter Lab** – experimentation and testing  
- **CUDA** – GPU acceleration
