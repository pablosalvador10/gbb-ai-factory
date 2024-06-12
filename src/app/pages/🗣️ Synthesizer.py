import streamlit as st
import dotenv
import asyncio
from utils.ml_logging import get_logger
from src.extractors.blob_data_extractor import AzureBlobDataExtractor
from src.aoai.azure_openai import AzureOpenAIManager
from src.ocr.document_intelligence import AzureDocumentIntelligenceManager
from src.utilsfunc import save_uploaded_file
from src.app.outputformatting import markdown_to_docx
import json
import tiktoken
import os

# Load environment variables
dotenv.load_dotenv(".env")

# Set up logger
logger = get_logger()

# Initialize session state variables if they don't exist
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'ai_response' not in st.session_state:
    st.session_state.ai_response = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hey, this is your AI assistant. Please look at the AI request submit and let's work together to make your content shine!"}]

# Accessing the session state variables
azure_openai_key = st.session_state.get('AZURE_OPENAI_KEY')
azure_aoai_chat_model_name_deployment_id = st.session_state.get('AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID')
azure_openai_api_endpoint = st.session_state.get('AZURE_OPENAI_API_ENDPOINT')
azure_openai_api_version = st.session_state.get('AZURE_OPENAI_API_VERSION')
azure_blob_storage = st.session_state.get('AZURE_BLOB_CONTAINER_NAME')
azure_document_intelligence_endpoint = st.session_state.get('AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT')
azure_document_intelligence_key = st.session_state.get('AZURE_DOCUMENT_INTELLIGENCE_KEY')
azure_storage_connection_string = st.session_state.get('AZURE_STORAGE_CONNECTION_STRING')

# Initialize managers if not already initialized
try:
    if 'document_intelligence_manager' not in st.session_state:
        st.session_state.document_intelligence_manager = AzureDocumentIntelligenceManager(
            azure_endpoint=azure_document_intelligence_endpoint, 
            azure_key=azure_document_intelligence_key
        )
    if 'blob_data_extractor_manager' not in st.session_state:
        st.session_state.blob_data_extractor_manager = AzureBlobDataExtractor(
            connect_str=azure_storage_connection_string, 
            container_name=azure_blob_storage
        )
    if 'azure_openai_manager' not in st.session_state:
        st.session_state.azure_openai_manager = AzureOpenAIManager(
            api_key=azure_openai_key,
            azure_endpoint=azure_openai_api_endpoint,
            api_version=azure_openai_api_version,
            chat_model_name=azure_aoai_chat_model_name_deployment_id
        )
except Exception as e:
    st.error(f"An error occurred: {e}. Please check your environment variables.")

# Sidebar layout for initial submission
with st.sidebar:
    st.title("🤖 AI RequestGPT")
    with st.expander("How to make a request", expanded=False):
        st.write("""
        This app allows you to submit a request to the AI model for generating a detailed, user-friendly documentation based on multiple documents and complex topics. The AI model will provide a response that you can review and download as a DOCX file.
        
        Follow these steps to make a request:
        1. Upload the documents you want the AI to analyze. You can upload multiple documents of types PNG, JPG, JPEG, and PDF.
        2. Enter your instructions for the AI in the 'Enter your instructions' field. Be as specific as possible to get the best results.
        3. Enter the topic in the 'Enter the topic' field. This will guide the AI in generating the documentation.
        4. Once you have filled all the fields and uploaded the documents, click on the 'Submit to AI' button to submit your request.
        5. Wait for the AI to process your request. Once done, you can review the generated documentation and download it as a DOCX file.
        """)

    uploaded_files = st.sidebar.file_uploader(
        "Upload documents", 
        type=["png", "jpg", "jpeg", "pdf"], 
        accept_multiple_files=True, 
        help="Upload the documents you want the AI to analyze. You can upload multiple documents of types PNG, JPG, JPEG, and PDF."
    )

    user_inputs = st.sidebar.text_input(
        "Enter your instructions", 
        help="Enter your instructions for the AI. Be as specific as possible to get the best results. For example, 'Summarize the key points in these documents.'"
    )

    topic = st.sidebar.text_input(
        "Enter the topic", 
        help="Enter the topic that will guide the AI in generating the documentation. For example, 'Machine Learning.'"
    )
    
    submit_to_ai = st.sidebar.button("Submit to AI", help="Submit your request to the AI model for generating documentation.")  


# Function to generate AI response
async def generate_ai_response(user_query):
    try:
        with st.spinner("🤖 Thinking..."):
            ai_response = await asyncio.to_thread(
                st.session_state.azure_openai_manager.generate_chat_response,
                conversation_history=st.session_state.conversation_history,
                system_message_content='''You are tasked with creating detailed, user-friendly documentation based on multiple documents and complex topics. 
                The goal is to distill this information into an easy-to-follow "How-To" guide. This documentation should 
                be structured with clear headings, subheadings, and step-by-step instructions that guide the user through 
                the necessary processes or concepts. Each section should be well-organized and written in simple language 
                to ensure that the content is accessible and understandable to users with varying levels of expertise. 
                The documentation should cover the setup, configuration, and usage of tools or techniques, 
                including practical examples and troubleshooting tips to address common issues or challenges that users 
                might encounter.''',
                query=user_query,
                max_tokens=3000
            )
        st.balloons()
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
        file_name='chat_history.json',
        mime='application/json',
        key='download-chat-history'
    )

# Function to download AI response as DOCX or PDF
def download_ai_response_as_docx_or_pdf():
    try:
        # Convert AI response from markdown to DOCX
        doc_io = markdown_to_docx(st.session_state.ai_response)
        
        # Let the user select the file format
        file_format = st.selectbox('Select file format', ['DOCX', 'PDF'])
        
        if file_format == 'DOCX':
            st.download_button(
                label="📁 Download .docx",
                data=doc_io,
                file_name='AI_Generated_Guide.docx',
                mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                key='download-docx'
            )
        elif file_format == 'PDF':
            # Convert the DOCX to PDF (you need to implement this function)
            st.download_button(
                label="📁 Download .pdf",
                data=doc_io,
                file_name='AI_Generated_Guide.pdf',
                mime='application/pdf',
                key='download-pdf'
            )
    except Exception as e:
        logger.error(f"Error generating {file_format} file: {e}")
        st.error(f"❌ Error generating {file_format} file. Please check the logs for more details.")


# Function to process a single file
async def process_single_file(semaphore, uploaded_file):
    async with semaphore:
        file_path = save_uploaded_file(uploaded_file)
        try:
            blob_name = os.path.basename(file_path)
            blob_url = await asyncio.to_thread(
                st.session_state.blob_data_extractor_manager.upload_file_to_blob, file_path, blob_name)
            result_ocr = await asyncio.to_thread(
                st.session_state.document_intelligence_manager.analyze_document,
                document_input=blob_url,
                model_type="prebuilt-layout",
                output_format="markdown",
                features=["OCR_HIGH_RESOLUTION"]
            )
            st.toast(f"Document '{uploaded_file.name}' has been successfully processed.", icon="😎")
            return result_ocr.content
        except Exception as e:
            logger.error(f"Error processing file {uploaded_file.name}: {e}")
            st.toast(f"Error processing file {uploaded_file.name}. Please check the logs for more details.")
            return ""

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

    st.toast(f"The processed content has {token_count} tokens.", icon="📊")

    query = f'''Given the content extracted from various documents using Optical Character Recognition (OCR) technology and provided in markdown format, your task is to create a high-quality, detailed "How-To" guide. The guide should distill complex topics into accessible, step-by-step instructions tailored for users seeking to understand or implement specific processes or concepts.
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
        - minimum length of the document should be {max_tokens} tokens'''

    st.session_state.conversation_history.append({"role": "user", "content": query})
    ai_response = await generate_ai_response(query)
    
    st.session_state['ai_response'] = ai_response
    st.session_state.chat_history.append({"role": "ai", "content": ai_response})

# Process submission
if submit_to_ai:
    if not uploaded_files or not user_inputs or not topic:
        st.sidebar.error("Please fill in all the fields and upload a document.")
    else:
        asyncio.run(process_and_generate_response(uploaded_files, user_inputs, topic))

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
        st.markdown("### 📥 Download Options")
        st.markdown("<small>Click the buttons below to download the AI response or chat history:</small>", unsafe_allow_html=True)
        download_ai_response_as_docx_or_pdf()
        download_chat_history()
