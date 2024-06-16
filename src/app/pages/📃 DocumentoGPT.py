import asyncio
import json
import os
import tempfile

import dotenv
import streamlit as st
import tiktoken

from src.aoai.azure_openai import AzureOpenAIManager
from src.app.outputformatting import markdown_to_docx
from src.extractors.blob_data_extractor import AzureBlobDataExtractor
from src.ocr.document_intelligence import AzureDocumentIntelligenceManager
from utils.ml_logging import get_logger

# Load environment variables
dotenv.load_dotenv(".env")

# Set up logger
logger = get_logger()

# Initialize session state variables if they don't exist
session_vars = ["conversation_history", "ai_response", "chat_history", "messages"]
initial_values = {
    "conversation_history": [],
    "ai_response": "",
    "chat_history": [],
    "messages": [
        {
            "role": "assistant",
            "content": "Hey, this is your AI assistant. Please look at the AI request submit and let's work together to make your content shine!",
        }
    ],
}
for var in session_vars:
    if var not in st.session_state:
        st.session_state[var] = initial_values.get(var, None)

st.set_page_config(
    page_title="DocumentoGPT",
    page_icon="📃",
)

# Check if environment variables have been loaded
if not st.session_state.get("env_vars_loaded", False):
    st.session_state.update({
        "azure_openai_manager": None,
        "document_intelligence_manager": None,
        "blob_data_extractor_manager": None,
        "env_vars_load_count_free": st.session_state.get("env_vars_load_count_free", 0) + 1
    })

    if st.session_state["env_vars_load_count_free"] <= 3:
        env_vars = {
            "AZURE_OPENAI_KEY": os.getenv("AZURE_OPENAI_KEY"),
            "AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID": os.getenv("AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID"),
            "AZURE_OPENAI_API_ENDPOINT": os.getenv("AZURE_OPENAI_API_ENDPOINT"),
            "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
            "AZURE_BLOB_CONTAINER_NAME": os.getenv("AZURE_BLOB_CONTAINER_NAME"),
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT": os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
            "AZURE_DOCUMENT_INTELLIGENCE_KEY": os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"),
            "AZURE_STORAGE_CONNECTION_STRING": os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
            "AZURE_AOAI_WHISPER_MODEL_DEPLOYMENT_ID": os.getenv("AZURE_AOAI_WHISPER_MODEL_DEPLOYMENT_ID"),
        }
        st.session_state.update(env_vars)

        # Initialize managers
        st.session_state["document_intelligence_manager"] = AzureDocumentIntelligenceManager(
            azure_endpoint=st.session_state["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"],
            azure_key=st.session_state["AZURE_DOCUMENT_INTELLIGENCE_KEY"],
        )
        st.session_state["blob_data_extractor_manager"] = AzureBlobDataExtractor(
            connect_str=st.session_state["AZURE_STORAGE_CONNECTION_STRING"],
            container_name=st.session_state["AZURE_BLOB_CONTAINER_NAME"],
        )
        st.session_state["azure_openai_manager"] = AzureOpenAIManager(
            api_key=st.session_state["AZURE_OPENAI_KEY"],
            azure_endpoint=st.session_state["AZURE_OPENAI_API_ENDPOINT"],
            api_version=st.session_state["AZURE_OPENAI_API_VERSION"],
            chat_model_name=st.session_state["AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID"],
        )
        st.toast(
            f'Free trial: {3 - st.session_state["env_vars_load_count_free"]} runs left. Please visit the main page and update your environment variables for unlimited runs.',
            icon="😎",
        )
else:
    try:
        # Reinitialize managers if necessary
        if "document_intelligence_manager" not in st.session_state:
            st.session_state["document_intelligence_manager"] = AzureDocumentIntelligenceManager(
                azure_endpoint=st.session_state["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"],
                azure_key=st.session_state["AZURE_DOCUMENT_INTELLIGENCE_KEY"],
            )
        if "blob_data_extractor_manager" not in st.session_state:
            st.session_state["blob_data_extractor_manager"] = AzureBlobDataExtractor(
                connect_str=st.session_state["AZURE_STORAGE_CONNECTION_STRING"],
                container_name=st.session_state["AZURE_BLOB_CONTAINER_NAME"],
            )
        if "azure_openai_manager" not in st.session_state:
            st.session_state["azure_openai_manager"] = AzureOpenAIManager(
                api_key=st.session_state["AZURE_OPENAI_KEY"],
                azure_endpoint=st.session_state["AZURE_OPENAI_API_ENDPOINT"],
                api_version=st.session_state["AZURE_OPENAI_API_VERSION"],
                chat_model_name=st.session_state["AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID"],
                whisper_model_name=st.session_state["AZURE_AOAI_WHISPER_MODEL_DEPLOYMENT_ID"],
            )
    except Exception as e:
        st.error(
            f"An error occurred: {str(e)}. Please ensure all environment variables are correctly set and accessible. If necessary, visit the introduction main page to update your environment variables."
        )

# Main layout for initial submission
with st.expander("What Can I Do? 🤔", expanded=False):
    st.markdown(
        """
        Here are some of the powerful features you can use:

        - **Summarize Documents, Images, and Audios**: Upload your files, and our application will provide you with a concise summary, saving you time and effort. This feature supports a variety of formats including text documents (.docx, .pdf), PowerPoint presentations (.pptx), images with text, and audio files.
        - **Generate Documentation**: Need to compile complex topics or multiple documents into a single, user-friendly document? Our application can generate comprehensive and easy-to-understand documentation, including executive memos, technical guides, and PowerPoint presentations.
        - **Translation**: Our application supports translations of documents and audio files from one language to another across multiple formats, helping you break down language barriers.
        - **Extract Insights and Information from Documents**: Upload your documents, and our application will analyze them to extract key insights and specific information such as names, dates, places, and more. This feature can help you quickly understand the main points and find the information you need without reading through the entire document.
        - **Interact with Complex Documents**: Allows you to interact with complex documents in a chat-like interface. Ask questions, and the application will provide answers based on the content of the document.
        - **Create Personalized Documents**: Our application enables the creation of personalized documents based on your input or templates. Whether it's generating a custom report, a business proposal, or a research paper, we've got you covered.

        To get started, please select an option from the sidebar. We're here to make your tasks easier! 👍
        """
    )

# Sidebar layout for initial submission
with st.sidebar:
    st.markdown("")
    operation = st.selectbox(
        "🎯 What would you like to accomplish today?",
        (
            "Generate Documentation",
            "Translation",
            "Summarization",
            "Extract Insights and Information",
        ),
        help="Select the type of operation you want to perform with the AI model.",
        placeholder="Generate Documentation",
    )

    if operation == "Generate Documentation":
        st.markdown(
            """
            <div style="text-align:center; font-size:30px; margin-top:10px;">
                ...
            </div>
            <div style="text-align:center; margin-top:20px;">
        """,
            unsafe_allow_html=True,
        )
        with st.expander("How-to guide 🚀", expanded=False):
            st.markdown(
                """
                Struggling with complex topics or juggling multiple documents? No worries! 🙌 Our application is here to transform your chaos into a single, user-friendly document. 📄✨

                Here's how it works:
                1. **Upload your documents**: These could be text files, PDFs, images with text, or even audio files. Our AI is versatile! 📚🎧
                2. **Provide your instructions**: Tell the AI what you need in clear terms. The more specific you are, the better the results! 💡
                3. **Sit back and relax**: Hit the 'Submit' button and let the AI do its magic. You'll get a comprehensive, easy-to-understand guide in no time! 🎩✨

                Ready to give it a try? Let's get started! 🚀
                """
            )

            uploaded_files = st.sidebar.file_uploader(
                "Upload documents",
                type=[
                    "png",
                    "jpg",
                    "jpeg",
                    "pdf",
                    "ppt",
                    "pptx",
                    "doc",
                    "docx",
                    "mp3",
                    "wav",
                ],
                accept_multiple_files=True,
                help="Upload the documents you want the AI to analyze. You can upload multiple documents of types PNG, JPG, JPEG, and PDF.",
            )

        document_type = st.sidebar.selectbox(
            "Select the type of document you are looking for",
            options=["How-to Guide", "Reference Manual", "API Documentation", "Other"],
            help="Choose the type of document you need. If 'Other', please provide details in the instructions box.",
        )

        if document_type == "Other":
            other_document_type = st.sidebar.text_input(
                "Specify the type of document",
                help="Describe the type of document you are looking for. For example, 'Marketing Plan', 'Research Paper', 'Business Proposal', etc.",
            )
        document_focus_areas = st.sidebar.text_area(
            "Document Focus Areas",
            value="N/A",
            help="Detailing the topics, questions, and areas you want the document to cover helps in creating content that meets your exact needs. Entering 'N/A' allows the AI to select topics for you.",
        )
        has_predefined_format = st.sidebar.radio(
            "Do you have a predefined output format?",
            options=["Yes", "No"],
            help="Select 'Yes' if you have a specific format in mind and would like to upload a document as a template.",
        )
        if has_predefined_format == "Yes":
            uploaded_file = st.sidebar.file_uploader(
                "Upload your document template",
                type=['pdf', 'docx', 'txt'],
                help="Upload the document that will serve as a template for the output format.",
            )

    submit_to_ai = st.sidebar.button("Submit to AI")


# Function to generate AI response
async def generate_ai_response(user_query):
    try:
        with st.spinner("🤖 Thinking..."):
            ai_response, _ = await asyncio.to_thread(
                st.session_state.azure_openai_manager.generate_chat_response,
                conversation_history=st.session_state.conversation_history,
                system_message_content="""You are tasked with creating detailed, 
                user-friendly documentation based on multiple documents and complex topics.
                The goal is to distill this information into an easy-to-follow "How-To" guide. 
                This documentation should be structured with clear headings, subheadings,
                and step-by-step instructions that guide the user through the necessary processes or concepts. 
                Each section should be well-organized and written in simple language to ensure that the content 
                is accessible and understandable to users with varying levels of expertise. 
                The documentation should cover the setup, configuration, and usage of tools or 
                techniques, including practical examples and troubleshooting tips to address common
                  issues or challenges that users might encounter.""",
                query=user_query,
                max_tokens=3000,
            )
        st.balloons()
        if st.session_state.get("env_vars_loaded", False):
            st.session_state["env_vars_load_count_free"] += 1
            st.toast(
                f'Free trial: {3 - st.session_state["env_vars_load_count_free"]} runs left. Please visit the main page and update your environment variables for unlimited runs.',
                icon="😎",
            )
        return ai_response
    except Exception as e:
        st.error(f"An error occurred while generating the AI response: {e}")
        return None


# Function to download chat history
def download_chat_history():
    chat_history_json = json.dumps(st.session_state.messages, indent=2)
    st.download_button(
        label="📜 Download Chat",
        data=chat_history_json,
        file_name="chat_history.json",
        mime="application/json",
        key="download-chat-history",
    )


# Function to download AI response as DOCX or PDF
def download_ai_response_as_docx_or_pdf():
    try:
        # Convert AI response from markdown to DOCX
        doc_io = markdown_to_docx(st.session_state.ai_response)

        # Let the user select the file format
        file_format = st.selectbox("Select file format", ["DOCX", "PDF"])

        if file_format == "DOCX":
            st.download_button(
                label="📁 Download .docx",
                data=doc_io,
                file_name="AI_Generated_Guide.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="download-docx",
            )
        elif file_format == "PDF":
            # Convert the DOCX to PDF (you need to implement this function)
            st.download_button(
                label="📁 Download .pdf",
                data=doc_io,
                file_name="AI_Generated_Guide.pdf",
                mime="application/pdf",
                key="download-pdf",
            )
    except Exception as e:
        logger.error(f"Error generating {file_format} file: {e}")
        st.error(f"❌ Error generating {file_format} file. Please check the logs for more details.")


async def process_single_file(semaphore, uploaded_file):
    async with semaphore:
        try:
            mime_type = uploaded_file.type
            logger.info(f"The MIME type is {mime_type}")
            if mime_type.startswith("audio/"):
                # TODO: Improve audio processing
                file_bytes = uploaded_file.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_file.write(file_bytes)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                    temp_file_path = temp_file.name
                    logger.info(f"Temporary file created at {temp_file_path}")
                    try:
                        result_ocr = await asyncio.to_thread(
                            st.session_state.azure_openai_manager.transcribe_audio_with_whisper,
                            audio_file_path=temp_file_path,
                        )
                    except Exception as e:
                        logger.error(f"Error during transcription: {e}")
            elif mime_type in ["image/png", "image/jpg", "image/jpeg"]:
                file_bytes = uploaded_file.read()
                result_ocr, _ = await asyncio.to_thread(
                    st.session_state.azure_openai_manager.generate_chat_response,
                    system_message_content="""You are an expert OCR AI model. Please analyze the image and provide a detailed summary.""",
                    query="""Focus on the details and make sure you extract all the details and return a detailed write-up of the content of the image""",
                    conversation_history=[],
                    image_bytes=[file_bytes],
                    stream=False,
                    max_tokens=1000,
                )
            elif mime_type in ["application/pdf"]:
                file_bytes = uploaded_file.read()
                result_ocr = await asyncio.to_thread(
                    st.session_state.document_intelligence_manager.analyze_document,
                    document_input=file_bytes,
                    model_type="prebuilt-layout",
                    output_format="markdown",
                    features=["OCR_HIGH_RESOLUTION"],
                )
                result_ocr = result_ocr.content
            else:
                file_bytes = uploaded_file.read()
                result_ocr = await asyncio.to_thread(
                    st.session_state.document_intelligence_manager.analyze_document,
                    document_input=file_bytes,
                    model_type="prebuilt-layout",
                    output_format="markdown",
                )
                result_ocr = result_ocr.content

            st.toast(f"Document '{uploaded_file.name}' has been successfully processed.", icon="😎")
            return result_ocr

        except Exception as e:
            logger.error(f"Error processing file {uploaded_file.name}: {e}")
            st.toast(f"Error processing file {uploaded_file.name}. Please check the logs for more details.")
            return ""
        finally:
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info("Temporary file removed successfully.")
                except Exception as e:
                    logger.error(f"Failed to remove temporary file: {e}")


# Function to process uploaded files and generate initial AI response
async def process_and_generate_response(uploaded_files, user_inputs, topic, max_tokens=3000):
    markdown_content = ""
    semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent tasks

    with st.spinner("🤖 Processing uploaded files..."):
        tasks = [process_single_file(semaphore, uploaded_file) for uploaded_file in uploaded_files]
        results = await asyncio.gather(*tasks)

    for result in results:
        if result:
            markdown_content += result + "\n\n"

    enc = tiktoken.get_encoding("cl100k_base")
    token_count = len(enc.encode(markdown_content))

    st.toast(f"The processed content has total of {token_count} tokens.", icon="📊")

    query = f"""Given the content extracted from various documents using Optical Character Recognition (OCR) technology and provided in markdown format, your task is to create a high-quality, detailed "How-To" guide. The guide should distill complex topics into accessible, step-by-step instructions tailored for users seeking to understand or implement specific processes or concepts.
        userinputs: {user_inputs}
        context: {markdown_content}

        Essential Steps for Crafting the Guide:

        1. **Content Synthesis**: Begin by synthesizing the OCR-extracted content. Identify crucial themes, technical concepts, and actionable instructions relevant to Copilot X and productivity enhancement. This synthesis forms the foundation of your guide's structure and content focus.

        2. **Target Audience Clarification**: Clearly define the guide's target audience. Understanding the audience's technical background, familiarity with Copilot X, and productivity goals is essential for customizing the guide's complexity and instructional style.

        3. **Structured Outline Development**: Construct a structured outline to organize the guide into coherent sections and subsections. Each section should concentrate on distinct aspects of using Copilot X for productivity, ensuring a logical progression from introductory concepts to advanced applications.

        4. **Guide Composition**:
            a. **Introduction**: Craft an introduction that outlines the guide's objectives, the significance of Copilot X for productivity, and what the readers will gain.
            b. **Detailed Instructions**: Following the outline, elaborate on each section with clear, technical instructions. Incorporate step-by-step processes, code snippets, examples, and best practices specific to Copilot X.
            c. **Conclusion**: Summarize the key takeaways, suggest further reading or resources, and encourage steps for practical application.

        5. **Comprehensive Review and Enhancement**: Thoroughly review the guide to ensure technical accuracy, clarity, and completeness. Revise any sections as necessary, and consider peer or expert feedback for additional insights.

        6. **Final Formatting and Release**: Apply professional formatting to enhance readability and visual appeal. Use diagrams, screenshots, or videos where applicable. Publish the guide in a format accessible to your target audience, ensuring it's ready for distribution and application.

        Additional Guidelines:

        - Begin with a clear agenda and systematically develop content within designated sections.
        - Employ straightforward language while explaining technical details, using examples to demystify complex concepts.
        - Dedicate ample time to crafting high-quality content, prioritizing accuracy and user engagement.
        - Base the guide explicitly on the OCR content and the nuanced requirements of the user's query regarding {topic}.
        - minimum length of the document should be {max_tokens} tokens"""

    st.session_state.conversation_history.append({"role": "user", "content": query})
    ai_response = await generate_ai_response(query)

    st.session_state["ai_response"] = ai_response
    st.session_state.chat_history.append({"role": "ai", "content": ai_response})


# Process submission
if submit_to_ai:
    if not uploaded_files:
        st.sidebar.error("Please fill in all the fields and upload a document.")
    else:
        asyncio.run(process_and_generate_response(uploaded_files))

# Chat interface for feedback and refining output
if st.session_state.ai_response:
    st.markdown("## AI Response")
    st.markdown(st.session_state.ai_response, unsafe_allow_html=True)
    feedback_prompt = st.chat_input("Enter your feedback or additional instructions:")

    if feedback_prompt:
        st.session_state.messages.append({"role": "user", "content": feedback_prompt})
        with st.chat_message("user"):
            st.markdown(feedback_prompt)

        ai_response = asyncio.run(generate_ai_response(feedback_prompt))
        st.session_state.ai_response = ai_response
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

        st.markdown("### Updated AI Response")
        st.markdown(ai_response, unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display download buttons in the sidebar if there is an AI response
if st.session_state.ai_response:
    with st.sidebar:
        st.markdown("<hr/>", unsafe_allow_html=True)
        with st.expander("📥 Download Center", expanded=False):
            download_ai_response_as_docx_or_pdf()
            download_chat_history()

st.sidebar.write(
    """
    <div style="text-align:center; font-size:30px; margin-top:10px;">
        ...
    </div>
    <div style="text-align:center; margin-top:20px;">
        <a href="https://github.com/pablosalvador10" target="_blank" style="text-decoration:none; margin: 0 10px;">
            <img src="https://img.icons8.com/fluent/48/000000/github.png" alt="GitHub" style="width:40px; height:40px;">
        </a>
        <a href="https://www.linkedin.com/in/pablosalvadorlopez/?locale=en_US" target="_blank" style="text-decoration:none; margin: 0 10px;">
            <img src="https://img.icons8.com/fluent/48/000000/linkedin.png" alt="LinkedIn" style="width:40px; height:40px;">
        </a>
        <a href="https://pablosalvador10.github.io/GenAI-Engineering-Diary/" target="_blank" style="text-decoration:none; margin: 0 10px;">
            <img src="https://img.icons8.com/?size=100&id=23438&format=png&color=000000" alt="Blog" style="width:40px; height:40px;">
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
