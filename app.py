import streamlit as st
from PIL import Image
from io import BytesIO
from streamlit_image_coordinates import streamlit_image_coordinates
import random
import torch  
from sam_inference import *
from sd_inference import *
import json
import base64


# Function to pad image to square
def pad_to_square(image):
    width, height = image.size
    max_dim = max(width, height)
    new_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))  # White background
    new_image.paste(image, ((max_dim - width) // 2, (max_dim - height) // 2))
    return new_image

# Initialize session state variables
if "clicked_point" not in st.session_state:
    st.session_state.clicked_point = None  # Store the single user click
if "input_points_display" not in st.session_state:
    st.session_state.input_points_display = []
if "mask_generated" not in st.session_state:
    st.session_state.mask_generated = False
if "output_generated" not in st.session_state:
    st.session_state.output_generated = False
if "mask" not in st.session_state:
    st.session_state.mask = None


# Streamlit UI setup
st.title("AI Inpainting App")
st.markdown("Upload an image, provide a text prompt, and watch AI magic in action!")

# Sidebar for example generated images
st.sidebar.markdown("### Example Generated Images")
try:
    example_image1 = Image.open("test images/generated1.jpg")
    example_image2 = Image.open("test images/generated2.jpg")
    example_image3 = Image.open("test images/generated3.png")

    st.sidebar.image(example_image1, caption="a cat walking on the streets, dim lighting", use_container_width=True)
    st.sidebar.image(example_image2, caption="a car driving on Mars. Studio lights, 1970s", use_container_width=True)
    st.sidebar.image(example_image3, caption="a man walking on snowy mountain. Natural lighting", use_container_width=True)
except FileNotFoundError:
    st.sidebar.error("Example images not found. Please ensure 'generated1.jpg' and 'generated2.jpg' are in the 'test images' folder.")

# Separate section for input image
st.markdown("### Upload Input Image")

uploaded_image = st.file_uploader("Drag and drop or click to upload", type=["png", "jpg", "jpeg"])

if uploaded_image:
    raw_image = Image.open(uploaded_image).convert("RGB")
    padded_image = pad_to_square(raw_image).resize((512, 512))

    st.markdown("**Click on the image below to select a point.**")

    # Get x, y coordinates using streamlit_image_coordinates
    coords = streamlit_image_coordinates(
        padded_image,  # Pass the Pillow Image object
        key="input_image"
    )

    # Compare new coords with existing clicked_point to decide if a new click occurred
    if coords is not None:
        new_click = (int(coords["x"]), int(coords["y"]))
        if new_click != st.session_state.clicked_point:
            st.session_state.clicked_point = new_click  # Update the selected point
            # Convert tuple to list to ensure correct format
            st.session_state.input_points_display = [list(st.session_state.clicked_point)]
            st.session_state.mask_generated = False
            st.session_state.output_generated = False

    # Display selected point(s)
    if st.session_state.input_points_display:
        st.write(f"Input Points: {st.session_state.input_points_display}")

    # Button to clear the selected point
    if st.button("Clear Point", key="clear_point"):
        st.session_state.clicked_point = None
        st.session_state.input_points_display = []
        st.session_state.mask_generated = False
        st.session_state.output_generated = False
        st.session_state.mask = None

else:
    # Reset session state if no image is uploaded
    st.session_state.clicked_point = None
    st.session_state.input_points_display = []
    st.session_state.mask_generated = False
    st.session_state.output_generated = False
    st.session_state.mask = None

# Load custom placeholder image
try:
    custom_placeholder = Image.open("placeholder.jpg").resize((512, 512))
except FileNotFoundError:
    st.error("Placeholder image not found. Please ensure 'placeholder.jpg' is in the working directory.")
    custom_placeholder = Image.new("RGB", (512, 512), (200, 200, 200))  # Gray placeholder

# Layout for mask and output sections
col1, col2 = st.columns(2, gap="medium")


# Input text for prompts
prompt = st.text_input("Inpainting Prompt", "a car driving on Mars. Studio lights, 1970s")
negative_prompt = st.text_input("Negative Prompt (Optional)", "artifacts, low quality, distortion")



with col1:
    st.markdown("### Mask")
    mask_placeholder = st.empty()
    if st.session_state.mask_generated and st.session_state.mask is not None:
        mask_rgb = Image.fromarray(mask_to_rgb(st.session_state.mask))
        mask_placeholder.image(mask_rgb, caption="Generated Mask", use_container_width=True)
    else:
        mask_placeholder.image(custom_placeholder, caption="Mask Not Available", use_container_width=True)

with col2:
    st.markdown("### Output")
    inpainted_placeholder = st.empty()
    if st.session_state.output_generated and st.session_state.mask is not None:

        payload = { 
            "prompt": prompt,
            "image": encode_img(padded_image, format='PNG'), 
            "mask_image": encode_img(Image.fromarray(mask_to_rgb(st.session_state.mask)), format='PNG'), 
            "num_inference_steps": 30,
            "guidance_scale": 7,
            "num_images_per_prompt": 1,
            "seed": random.randint(0, 2**32 - 1),
            "negative_prompt": negative_prompt,
        }

  
        query_response = call_sagemaker_inpaint(payload)
        response_dict = json.loads(query_response['Body'].read())
        generated_images = response_dict['generated_images']

        print(len(generated_images))
        with BytesIO(base64.b64decode(generated_images[0].encode())) as generated_image_decoded:
            with Image.open(generated_image_decoded) as generated_image_np:
                inpainted_image = generated_image_np.convert("RGB")


        inpainted_placeholder.image(inpainted_image, caption="Inpainted Image", use_container_width=True)
    else:
        inpainted_placeholder.image(custom_placeholder, caption="Output Not Available", use_container_width=True)

# Generate Mask Button
if uploaded_image and st.session_state.clicked_point:
    if not st.session_state.mask_generated:
        if st.button("Generate Mask", key="generate_mask"):
            try:
                st.write("Processing the image and generating the mask...")
                # Convert points to a 4D tensor: [batch_size, point_batch_size, nb_points_per_image, 2]
                input_points = torch.tensor(st.session_state.input_points_display, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                mask = get_processed_inputs(padded_image, input_points)
                mask_rgb = Image.fromarray(mask_to_rgb(mask))
                mask_placeholder.image(mask_rgb, caption="Generated Mask", use_container_width=True)
                st.session_state.mask = mask
                st.session_state.mask_generated = True
            except Exception as e:
                st.error(f"An error occurred while generating the mask: {str(e)}")

# Generate Output Image Button
if st.session_state.mask_generated and uploaded_image:
    if not st.session_state.output_generated:
        if st.button("Generate Output Image", key="generate_output"):
            try:
                st.write("Performing inpainting...")
                # Ensure mask is in the correct format
                if st.session_state.mask is None:
                    raise ValueError("Mask is not generated.")

                payload = { 
                    "prompt": prompt,
                    "image": encode_img(padded_image, format='PNG'), 
                    "mask_image": encode_img(Image.fromarray(mask_to_rgb(st.session_state.mask)), format='PNG'), 
                    "num_inference_steps": 30,
                    "guidance_scale": 7,
                    "num_images_per_prompt": 1,
                    "seed": random.randint(0, 2**32 - 1),
                    "negative_prompt": negative_prompt,
                }

                query_response = call_sagemaker_inpaint(payload)
                response_dict = json.loads(query_response['Body'].read())
                generated_images = response_dict['generated_images']

                print(len(generated_images))

                with BytesIO(base64.b64decode(generated_images[0].encode())) as generated_image_decoded:
                    with Image.open(generated_image_decoded) as generated_image_np:
                        inpainted_image = generated_image_np.convert("RGB")

                # Display the inpainted image
                inpainted_placeholder.image(inpainted_image, caption="Inpainted Image", use_container_width=True)
                st.session_state.output_generated = True

                # Option to download the inpainted image
                from io import BytesIO
                inpainted_buffer = BytesIO()
                inpainted_image.save(inpainted_buffer, format="PNG")
                inpainted_buffer.seek(0)

                download_clicked = st.download_button(
                    label="Download Inpainted Image",
                    data=inpainted_buffer,
                    file_name="inpainted_image.png",
                    mime="image/png"
                )

                # Reset state after download
                if download_clicked:
                    st.success("Image downloaded successfully. Resetting to start a new process.")
                    st.session_state.mask_generated = False
                    st.session_state.output_generated = False
                    st.session_state.mask = None
                    st.session_state.clicked_point = None
                    st.session_state.input_points_display = []
                    # Clear placeholders
                    mask_placeholder.image(custom_placeholder, caption="Mask Not Available", use_container_width=True)
                    inpainted_placeholder.image(custom_placeholder, caption="Output Not Available", use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred while generating the output image: {str(e)}")

    else:
        if st.button("Revert to Generate Mask", key="revert_mask"):
            st.session_state.mask_generated = False
            st.session_state.output_generated = False
            st.session_state.mask = None
            mask_placeholder.image(custom_placeholder, caption="Mask Not Available", use_container_width=True)
