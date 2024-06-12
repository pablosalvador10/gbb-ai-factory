import base64
import streamlit as st

from src.extractors.blob_data_extractor import AzureBlobDataExtractor
from src.aoai.azure_openai import AzureOpenAIManager
from src.ocr.document_intelligence import AzureDocumentIntelligenceManager

# Function to convert image to base64
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

st.set_page_config(
    page_title="eLearningFactoryGPT",
    page_icon="🎓",
)

# Initialize environment variables in session state
env_vars = {
    'AZURE_OPENAI_KEY': '',
    'AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID': '',
    'AZURE_OPENAI_API_ENDPOINT': '',
    'AZURE_OPENAI_API_VERSION': '',
    'AZURE_BLOB_CONTAINER_NAME': '',
    'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT': '',
    'AZURE_DOCUMENT_INTELLIGENCE_KEY': '',
    'AZURE_STORAGE_CONNECTION_STRING': ''
}

for var in env_vars:
    if var not in st.session_state:
        st.session_state[var] = env_vars[var]

# Sidebar inputs for environment variables
st.sidebar.header("Configure Azure Environment Variables")

with st.sidebar.expander("Required Environment Variables", expanded=False):
    st.markdown("""
    Please provide the following Azure environment variables to configure the application. You can find these details in the respective Azure services.

    - **Azure OpenAI Key**: Get your key from [Azure OpenAI](https://azure.microsoft.com/en-us/services/openai/)
    - **Chat Model Deployment ID**: Your deployment ID for the chat model in Azure OpenAI.
    - **Completion Model Deployment ID**: Your deployment ID for the completion model in Azure OpenAI.
    - **Embedding Deployment ID**: Your deployment ID for the embedding model in Azure OpenAI.
    - **API Endpoint**: The API endpoint for your Azure OpenAI service.
    - **API Version**: The version of the Azure OpenAI API you are using.
    - **Document Intelligence Endpoint**: The endpoint for your Azure Document Intelligence API.
    - **Document Intelligence Key**: Your key for the Azure Document Intelligence API.
    - **Storage Connection String**: Your Azure Storage connection string.
    """)

st.session_state['AZURE_OPENAI_KEY'] = st.sidebar.text_input(
    "Azure OpenAI Key", 
    value=st.session_state['AZURE_OPENAI_KEY'], 
    help="Enter your Azure OpenAI key.",
    type="password",
    placeholder="Enter your Azure OpenAI Key",
    label_visibility="visible"
)
st.session_state['AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID'] = st.sidebar.text_input(
    "Chat Model Deployment ID", 
    value=st.session_state['AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID'], 
    help="Enter your chat model deployment ID for Azure OpenAI.",
    placeholder="Enter your Chat Model Deployment ID",
    label_visibility="visible"
)

st.session_state['AZURE_OPENAI_API_ENDPOINT'] = st.sidebar.text_input(
    "API Endpoint", 
    value=st.session_state['AZURE_OPENAI_API_ENDPOINT'], 
    help="Enter the API endpoint for Azure OpenAI.",
    placeholder="Enter your API Endpoint",
    label_visibility="visible"
)
st.session_state['AZURE_OPENAI_API_VERSION'] = st.sidebar.text_input(
    "API Version", 
    value=st.session_state['AZURE_OPENAI_API_VERSION'], 
    help="Enter the API version for Azure OpenAI.",
    placeholder="Enter your API Version",
    label_visibility="visible"
)
st.session_state['AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT'] = st.sidebar.text_input(
    "Document Intelligence Endpoint", 
    value=st.session_state['AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT'], 
    help="Enter the endpoint for Azure Document Intelligence API.",
    placeholder="Enter your Document Intelligence Endpoint",
    label_visibility="visible"
)
st.session_state['AZURE_DOCUMENT_INTELLIGENCE_KEY'] = st.sidebar.text_input(
    "Document Intelligence Key", 
    value=st.session_state['AZURE_DOCUMENT_INTELLIGENCE_KEY'], 
    help="Enter your Azure Document Intelligence API key.",
    type="password",
    placeholder="Enter your Document Intelligence Key",
    label_visibility="visible"
)
st.session_state['AZURE_BLOB_CONTAINER_NAME'] = st.sidebar.text_input(
    "Blob Container Name", 
    value=st.session_state.get('CONTAINER_NAME', ''), 
    help="Enter your Azure Blob Storage container name.",
    placeholder="Enter your Blob Container Name",
    label_visibility="visible"
)
st.session_state['AZURE_STORAGE_CONNECTION_STRING'] = st.sidebar.text_input(
    "Storage Connection String", 
    value=st.session_state.get('AZURE_STORAGE_CONNECTION_STRING', ''), 
    help="Enter your Azure Storage connection string.",
    type="password",
    placeholder="Enter your Storage Connection String",
    label_visibility="visible"
)

if st.sidebar.button("Validate Environment Variables"):
    try:
        # Initialize managers if they don't exist in session state
        for manager_name, manager in [
            ("document_intelligence_manager", AzureDocumentIntelligenceManager(azure_endpoint=st.session_state['AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT'], 
                                                                               azure_key=st.session_state['AZURE_DOCUMENT_INTELLIGENCE_KEY'])),
            ("blob_data_extractor_manager", AzureBlobDataExtractor(connect_str=st.session_state['AZURE_STORAGE_CONNECTION_STRING'], 
                                                                   container_name=st.session_state['AZURE_BLOB_CONTAINER_NAME'])),
            ("azure_openai_manager", AzureOpenAIManager(api_key=st.session_state['AZURE_OPENAI_KEY'],
                                                        azure_endpoint=st.session_state['AZURE_OPENAI_API_ENDPOINT'],
                                                        api_version=st.session_state['AZURE_OPENAI_API_VERSION'],
                                                        chat_model_name=st.session_state['AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID']))]:
            if manager_name not in st.session_state:
                st.session_state[manager_name] = manager
        st.sidebar.success("All environment variables and managers have been successfully validated and initialized.")
    except Exception as e:
        st.sidebar.error(f"An error occurred: {e}. Please check your environment variables.")

# Web user interface
st.write(
    f"""
    <h1 style="text-align:center;">
        Welcome to eLearningFactoryGPT! 🎓
        <br>
        <span style="font-style:italic; font-size:0.7em;"> Powered by the AI Global Black Belt Team </span> <img src="data:image/png;base64,{get_image_base64('./utils/images/azure_logo.png')}" alt="logo" style="width:30px;height:30px;">
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    Welcome to **eLearningFactoryGPT**, your one-stop solution for automated eLearning content creation. Leveraging the power of Azure OpenAI, we provide a multimodal AI service designed to revolutionize the way educational content is created. 

    Our goal is to streamline the content creation process, reducing both the time and cost involved, while maintaining the highest quality standards. With eLearningFactoryGPT, you can focus on what truly matters - delivering exceptional learning experiences.

    ### Curious to Learn More?
    - Discover the power of [Azure OpenAI](https://azure.microsoft.com/en-us/services/openai/) and how it's changing the world of AI.
    - Dive into our comprehensive [documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/openai-service/) to understand how our service works.
    - Join the conversation in our [community forums](https://discuss.streamlit.io) where experts and enthusiasts discuss the latest trends and challenges in AI.

    #### Getting Started
    To get the most out of eLearningFactoryGPT, you'll need a few keys. Here's how you can obtain them:
    - **Azure OpenAI Key**: Sign up for Azure OpenAI and [get your key here](https://azure.microsoft.com/en-us/services/openai/).
    - **Azure Storage Connection String**: Follow the [Azure Storage documentation](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-keys-manage) to set up your storage and obtain your connection string.
    - **Document Intelligence API**: Get your API key and endpoint by following the [Azure Document Intelligence documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/form-recognizer/).

    Embark on your journey with eLearningFactoryGPT and transform the way you create content!
    """
)

