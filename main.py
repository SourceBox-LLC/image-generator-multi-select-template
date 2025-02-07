import logging
import requests
from dotenv import load_dotenv
import os, io
import random
import string
import uuid
from PIL import Image
import time
import streamlit as st


load_dotenv()


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# URLs for various Hugging Face models (Stability AI, Boreal, Flux, and Phantasma Anime)
stability_api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
boreal_api_url = "https://api-inference.huggingface.co/models/kudzueye/Boreal"
flux_api_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
phantasma_anime_api_url = "https://api-inference.huggingface.co/models/alvdansen/phantasma-anime"

# Prepare headers for API requests to Hugging Face, including the authorization token
hf_headers = {"Authorization": f"Bearer {api_token}"}

#uniqe image identifier
def random_sig():
    """Generates a 3-character random signature, which can be a combination of letters or digits."""
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(3))



def query_image(prompt, api_url, retries=3, delay=5):
    """
    Generic function to query a Hugging Face API for generating images based on a prompt.
    The function sends a POST request to the specified API URL with the prompt data.
    Returns the binary content of the generated image or None if an error occurred.
    """
    for attempt in range(retries):
        try:
            logging.info(f"Querying {api_url} with prompt: {prompt}")
            response = requests.post(api_url, headers=hf_headers, json={"inputs": prompt})
            response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
            logging.info(f"Received response with status code {response.status_code}")
            return response.content  # Return the image content as bytes
        except requests.exceptions.RequestException as e:
            logging.info(f"Error querying {api_url}: {e}")
            if attempt < retries - 1:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error("Max retries reached. Failed to generate image.")
                return None


def query_flux_image(prompt):
    return query_image(prompt, flux_api_url)

def query_boreal_image(prompt):
    return query_image(prompt, boreal_api_url)

def query_stability_image(prompt):
    return query_image(prompt, stability_api_url)

def query_phantasma_anime_image(prompt):
    return query_image(prompt, phantasma_anime_api_url)



def generate_image(prompt, generator):
    try:
        logging.info("Received request to generate image")
        logging.info(f"Prompt: {prompt}, Generator: {generator}")

        if not prompt or not generator:
            logging.error("Prompt and generator type are required.")
            return "Prompt and generator type are required."

        unique_prompt = f"{prompt} - {random_sig()}"
        logging.debug(f"Unique prompt: {unique_prompt}")

        image_bytes = None
        
        if generator == "flux":
            image_bytes = query_flux_image(unique_prompt)
        elif generator == "stability":
            image_bytes = query_stability_image(unique_prompt)
        elif generator == "boreal":
            image_bytes = query_boreal_image(unique_prompt)
        elif generator == "phantasma-anime":
            image_bytes = query_phantasma_anime_image(unique_prompt)
        else:
            logging.error("Invalid generator selected")
            return "error: Invalid generator selected"

        if not image_bytes:
            logging.error("Failed to generate image from the selected API.")
            return "error: Failed to generate image from the selected API."

        image_name = f"{generator}_image_{uuid.uuid4().hex}.png"
        image_path = f"static/{image_name}"
        logging.debug(f"Saving image to: {image_path}")

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.save(image_path)
            logging.info(f"Image saved successfully: {image_name}")
            return image_path  # Return the path of the saved image
        except Exception as e:
            logging.error(f"Error saving image: {e}")
            return "error: Error saving image"

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return str(e)
        
        
 

#________streamlit GUI________


# Set up the Streamlit app
st.title("Source Studio")
st.write("Generate images using different AI models by providing a prompt and selecting a generator.")

# Initialize session state for conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

# Input for the prompt
prompt = st.chat_input("Say something")

# Dropdown for selecting the generator
generator = st.selectbox(
    "Select a generator",
    ("flux", "stability", "boreal", "phantasma-anime")
)

# Automatically generate the image when a prompt is entered
if prompt:
    st.write(f"Prompt: {prompt}")
    result = generate_image(prompt, generator)
    
    if result.startswith("error"):
        st.error(result)
    else:
        # Add the prompt and result to the conversation history
        st.session_state.history.append((prompt, result))

# Display the conversation history
st.write("### Conversation History")
for i, (past_prompt, image_path) in enumerate(st.session_state.history):
    st.write(f"Prompt {i+1}: {past_prompt}")
    st.image(image_path, caption=f"Generated Image {i+1}")
