import os
from io import BytesIO
from typing import Any, Dict

import streamlit as st
from PIL import Image
from src.aoai.azure_openai import AzureOpenAIManager
import autogen
from src.autogen_helper.dallecritic import AzureDalleImageGenerator, ImageGeneration, extract_images
from autogen.agentchat.contrib import img_utils

from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

st.set_page_config(
    page_title="SketchGPT",
    page_icon="🖼️",
)

# Azure Open AI Completion Configuration
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID = os.getenv("AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID")
AZURE_OPENAI_API_ENDPOINT = os.getenv("AZURE_OPENAI_API_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_AOAI_KEY_VISION = os.getenv("AZURE_AOAI_KEY_VISION")
AZURE_AOAI_API_VERSION_VISION = os.getenv("AZURE_AOAI_API_VERSION_VISION")
AZURE_AOAI_API_ENDPOINT_VISION = os.getenv("AZURE_AOAI_API_ENDPOINT_VISION")
AZURE_AOAI_DALLE_MODEL_DEPLOYMENT_ID = os.getenv("AZURE_AOAI_DALLE_MODEL_DEPLOYMENT_ID")

# Initialize session state
if "env_vars_loaded" not in st.session_state:
    st.session_state.env_vars_loaded = False

if not st.session_state.env_vars_loaded:
    st.session_state.update(
        {
            "azure_openai_manager": None,
        }
    )

    env_vars = {
        "AZURE_OPENAI_KEY": AZURE_OPENAI_KEY,
        "AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID": AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID,
        "AZURE_OPENAI_API_ENDPOINT": AZURE_OPENAI_API_ENDPOINT,
        "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
        "AZURE_AOAI_KEY_VISION": AZURE_AOAI_KEY_VISION,
        "AZURE_AOAI_API_VERSION_VISION": AZURE_AOAI_API_VERSION_VISION,
        "AZURE_AOAI_API_ENDPOINT_VISION": AZURE_AOAI_API_ENDPOINT_VISION,
        "AZURE_AOAI_DALLE_MODEL_DEPLOYMENT_ID": AZURE_AOAI_DALLE_MODEL_DEPLOYMENT_ID,
    }
    st.session_state.update(env_vars)

    st.session_state["azure_openai_manager"] = AzureOpenAIManager(
        api_key=st.session_state["AZURE_OPENAI_KEY"],
        azure_endpoint=st.session_state["AZURE_OPENAI_API_ENDPOINT"],
        api_version=st.session_state["AZURE_OPENAI_API_VERSION"],
        chat_model_name=st.session_state["AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID"],
    )

    st.session_state["llm_config_gpt4o"] = {
        "config_list": [
            {
                "model": AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID,
                "api_type": "azure",
                "api_key": AZURE_OPENAI_KEY,
                "base_url": AZURE_OPENAI_API_ENDPOINT,
                "api_version": AZURE_OPENAI_API_VERSION
            }
        ]
    }

    st.session_state["llm_config_dalle3"] = {
        "config_list": [
            {
                "model": AZURE_AOAI_DALLE_MODEL_DEPLOYMENT_ID,
                "api_type": "azure",
                "api_key": AZURE_AOAI_KEY_VISION,
                "base_url": AZURE_AOAI_API_ENDPOINT_VISION,
                "api_version": AZURE_AOAI_API_VERSION_VISION
            }
        ]
    }
    st.session_state.env_vars_loaded = True

def _is_termination_message(msg) -> bool:
    if isinstance(msg.get("content"), str):
        return msg["content"].rstrip().endswith("TERMINATE")
    elif isinstance(msg.get("content"), list):
        for content in msg["content"]:
            if isinstance(content, dict) and "text" in content:
                return content["text"].rstrip().endswith("TERMINATE")
    return False

def critic_agent(CRITIC_SYSTEM_MESSAGE) -> autogen.ConversableAgent:
    return autogen.ConversableAgent(
        name="critic",
        llm_config=st.session_state['llm_config_gpt4o'],
        system_message=CRITIC_SYSTEM_MESSAGE,
        max_consecutive_auto_reply=3,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: _is_termination_message(msg),
    )

def image_generator_agent() -> autogen.ConversableAgent:
    agent = autogen.ConversableAgent(
        name="dalle",
        llm_config=st.session_state['llm_config_gpt4o'],
        max_consecutive_auto_reply=3,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: _is_termination_message(msg),
    )

    dalle_gen = AzureDalleImageGenerator(llm_config=st.session_state['llm_config_dalle3'])
    image_gen_capability = ImageGeneration(
        image_generator=dalle_gen, text_analyzer_llm_config=st.session_state['llm_config_gpt4o']
    )
    image_gen_capability.add_to_agent(agent)
    return agent

def process_submission(prompt: str, criteria: str):
    CRITIC_SYSTEM_MESSAGE = f"""
        To ensure the generated images meet our high-quality standards, we evaluate them based on the following criteria: color vibrancy, shape accuracy, text clarity, and overall composition. Below are the steps and standards for the evaluation and improvement process:

        **Quality Standards**:
        {criteria}

        **Evaluation and Improvement Instructions**:

        1. **CRITICS**: Identify areas where the image falls short of the above standards. Provide specific feedback, such as "The colors are too muted and need to be more vibrant to convey the intended mood."

        2. **PROMPT**: Based on your critique, offer a revised prompt that includes clear directions for improvement. For example, "Generate an image with bright and vibrant colors, sharp and accurate shapes, and clear, legible text, ensuring a balanced and harmonious composition."

        **Termination Condition**:
        - If the image satisfactorily meets all the quality standards, conclude the evaluation by responding with "TERMINATE". This indicates that the image has achieved the desired level of quality and no further adjustments are necessary.
        """
    
    dalle = image_generator_agent()
    critic = critic_agent(CRITIC_SYSTEM_MESSAGE)

    result = dalle.initiate_chat(critic, message=prompt)
    images = extract_images(dalle, critic)

    all_messages = dalle.chat_messages[critic]

    return images, all_messages

def pil_to_bytes(image: Image) -> bytes:
    with BytesIO() as output:
        image.save(output, format="PNG")
        return output.getvalue()

with st.sidebar:
    with st.expander("Add Required Environment Variables ⚙️", expanded=False):
        st.markdown("""
            Please provide the following Azure environment variables to configure the DALL-E model in the application. You can find these details in the Azure services.

            - **Azure OpenAI Key for Vision**: Your key for the Azure OpenAI service, specifically for vision tasks like DALL-E.
            - **DALL-E Model Deployment ID**: Your deployment ID for the DALL-E model in Azure OpenAI.
            - **API Endpoint for Vision**: The API endpoint for your Azure OpenAI service, specifically for vision tasks.
            - **API Version for Vision**: The version of the Azure OpenAI API you are using for vision tasks.
            """
        )
        st.session_state["AZURE_AOAI_KEY_VISION"] = st.text_input(
            "Azure OpenAI Key for Vision",
            value=st.session_state.get("AZURE_AOAI_KEY_VISION", ""),
            help="Enter your Azure OpenAI key for vision tasks.",
            type="password",
            placeholder="Enter your Azure OpenAI Key for Vision",
            label_visibility="visible",
        )
        st.session_state["AZURE_AOAI_DALLE_MODEL_DEPLOYMENT_ID"] = st.text_input(
            "DALL-E Model Deployment ID",
            value=st.session_state.get("AZURE_AOAI_DALLE_MODEL_DEPLOYMENT_ID", ""),
            help="Enter your DALL-E model deployment ID for Azure OpenAI.",
            placeholder="Enter your DALL-E Model Deployment ID",
            label_visibility="visible",
        )

        st.session_state["AZURE_AOAI_API_ENDPOINT_VISION"] = st.text_input(
            "API Endpoint for Vision",
            value=st.session_state.get("AZURE_AOAI_API_ENDPOINT_VISION", ""),
            help="Enter the API endpoint for Azure OpenAI vision tasks.",
            placeholder="Enter your API Endpoint for Vision",
            label_visibility="visible",
        )
        st.session_state["AZURE_AOAI_API_VERSION_VISION"] = st.text_input(
            "API Version for Vision",
            value=st.session_state.get("AZURE_AOAI_API_VERSION_VISION", ""),
            help="Enter the API version for Azure OpenAI vision tasks.",
            placeholder="Enter your API Version for Vision",
            label_visibility="visible",
        )
    st.markdown("### 🚀 Start Your Creative Journey")
    user_prompt = st.text_area(
        "Enter your text prompt",
        help="Describe the image or task you want to generate or accomplish. Be as specific as possible to get the best results.",
    )
    evaluation_criteria = st.text_area(
        "Enter your evaluation criteria",
        help="Specify the criteria that the critic agent should use to evaluate the generated image or task outcome. This could include aspects like creativity, relevance, accuracy, etc.",
    )
    submit_button = st.button("Submit")

def display_chat(messages):
    for message in messages:
        sender = message.get("sender", "Agent")
        content_list = message.get("content", [])
        for content in content_list:
            if isinstance(content, str):
                st.markdown(f"**{sender}:** {content}")
            elif content.get("type") == "image_url":
                img_url = content["image_url"]["url"]
                st.markdown(f"**{sender}:**")
                st.image(img_utils.get_pil_image(img_url), caption="Generated Image")

if submit_button and user_prompt and evaluation_criteria:
    with st.spinner("Generating image, please wait..."):
        try:
            images, all_messages = process_submission(user_prompt, evaluation_criteria)
            if images:
                st.success("Image generation completed successfully!")
                for idx, image in enumerate(reversed(images)):
                    st.image(image.resize((300, 300)), caption=f"Generated Image {idx+1}")
                    img_download = st.download_button(
                        label="Download Image",
                        data=pil_to_bytes(image),
                        file_name=f"generated_image_{idx+1}.png",
                        mime="image/png"
                    )
                #st.markdown("### Agent Conversation History")
                #display_chat(all_messages)
            else:
                st.error("Failed to generate images.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please enter a prompt and evaluation criteria, then click Submit.")
