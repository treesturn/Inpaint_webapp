


https://github.com/user-attachments/assets/0cf734a8-7394-4f8f-9ab9-21dba6071794



# Inpaint Webapp

This project is a web application that uses **Facebook’s Segment Anything Model (SAM)** to segment an input image 
and then applies an **inpainting diffusion model** to transform or replace the background. The front end is built with 
**Streamlit**, allowing users to interactively upload images and visualize the results in real-time. While the SAM model can run locally on a CPU, 
the inpainting diffusion model requires GPU acceleration. To address this, the project integrates with an AWS SageMaker endpoint 
to run the inpainting diffusion model efficiently.

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
**Do note that using the endpoint from AWS Sagemaker does incur additional cost.**

1. In you AWS Sagemaker studio, navigate to jumpstart and search "Inpaint". Select the Stable Diffusion 2 Inpainting FP16 option
   
![Screenshot 2025-03-19 195506](https://github.com/user-attachments/assets/84494a21-82eb-4df2-875c-9a05b491859e)

2. Deploy the model endpoint by clicking "Deploy" on the top right
 
![Screenshot 2025-03-19 195538](https://github.com/user-attachments/assets/f5695cc4-dccc-4d6f-9d75-8e83fac7f277)

2. Deploy the model endpoint on your chosen instance type. Once chosen, click deploy on the bottom right. 

![Screenshot 2025-03-19 195606](https://github.com/user-attachments/assets/7426a940-e82d-4d26-b989-fd310ba7824c)
