


https://github.com/user-attachments/assets/0cf734a8-7394-4f8f-9ab9-21dba6071794



# Inpaint Webapp

This project is a web application that uses **Facebook’s Segment Anything Model (SAM)** to segment an input image 
and then applies an **inpainting diffusion model** to transform or replace the background. The front end is built with 
**Streamlit**, allowing users to interactively upload images and visualize the results in real-time.

## Features

- **Image Upload**: Users can upload any image through the Streamlit interface.  
- **Segmentation**: Facebook’s SAM automatically segments the foreground from the background.  
- **Inpainting Diffusion**: The inpainting model modifies or replaces the segmented background with a new texture or design.  
- **Interactive UI**: Built with Streamlit for easy image display and control over the segmentation process.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/inpaint_webapp.git
   cd inpaint_webapp
2. **pip install requirements.txt**:
   ```bash
    pip install -r requirements.txt
2. **run app**:
   ```bash
    streamlit run app.py

## Inpaint Diffusion model setup from AWS Sagemaker

