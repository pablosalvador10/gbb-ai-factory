import base64
import streamlit as st


# Function to convert image to base64
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


st.set_page_config(
    page_title="eLearningFactoryGPT",
    page_icon="🎓",
)

# Web user interface
st.write(
    f"""
    <h1 style="text-align:center;">
        Welcome to eLearningFactoryGPT! 🎓
        <br>
        <span style="font-style:italic; font-size:0.7em;"> AI Global Black Belt Team </span> <img src="data:image/png;base64,{get_image_base64('./utils/images/azure_logo.png')}" alt="logo" style="width:30px;height:30px;">
    </h1>
    """,
    unsafe_allow_html=True,
)

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    eLearningGPT is an AI service powered by Azure OpenAI, offering multimodality capabilities. It aims to automate the eLearning content creation process, reducing the time and cost involved in creating high-quality educational content.
    
    ### Want to learn more?
    - Check out [Azure OpenAI](https://azure.microsoft.com/en-us/services/openai/)
    - Jump into our [documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/openai-service/)
    - Ask a question in our [community forums](https://discuss.streamlit.io)
    
    """
)