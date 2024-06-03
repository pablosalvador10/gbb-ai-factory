import base64
import streamlit as st
import re
import secrets
import string
from src.app.utilsapp import send_email
from typing import List, Dict
from dotenv import load_dotenv
import datetime

from src.app.qualiFictionAlgo import calculate_total_score
from src.ocr.transformer import GPT4VisionManager
from utils.ml_logging import get_logger
from src.utilsfunc import save_uploaded_file
from src.app.oumapping import ou_email_mapping
from src import settings
from src.ocr.cosmosDB_indexer import CosmosDBIndexer

# Load environment variables from .env file
load_dotenv()

# Set up logger
logger = get_logger()

DELAY_TIME = 0.01
FROM_EMAIL= "Pablosalvadorlopez@outlook.com"

# Initialize GPT4VisionManager in session state
if "gpt4_vision_manager" not in st.session_state:
    st.session_state.gpt4_vision_manager = GPT4VisionManager()

if "cosmos_manager" not in st.session_state:
    st.session_state.cosmos_client = CosmosDBIndexer(database_name="gbbai-qualifiction-db",
                                                     container_name="gbbai-qualifiction")


# Function to convert image to base64 for embedding
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
    
# Chatbot interface
st.title("💡 Submit Your Request !")

"""
🚀 Powered by Azure OpenAI. This AI Request Evaluator helps you analyze complex request data, make informed decisions, and provide detailed explanations for those decisions. 

In this example, we're using `GPT4 Turbo with vision` to demonstrate the decision-making capabilities of the AI Request Evaluator in an interactive Streamlit app.

Explore logic behind the decision-making AI at [github.com/gbbai-smart-qualify-engagement](https://github.com/pablosalvador10/gbbai-smart-qualify-engagement).
"""

# About App expander
with st.expander("About this App"):
    st.markdown(
        """
        ### 🌟 Application Overview
        This application demonstrates the power of Azure AI in a real-time decision-making context. It seamlessly integrates Azure AI services to provide a sophisticated request evaluation experience.

        #### Key Features:
        - **Request Analysis**: Utilizes Azure AI to understand complex relationships in your request data.
        - **Decision Making**: Employs Azure OpenAI GPT-4 for advanced decision-making based on your request data.
        - **Next Steps Suggestions**: Leverages Azure AI to provide suggestions for improvement and the steps that should be taken before the request can be reconsidered.

        ### 🔧 Prerequisites and Dependencies
        To fully experience this application, the following Azure services are required:
        - Azure OpenAI Service: Set up an instance and obtain an API key for access to GPT-4 capabilities.
        - Azure AI Service: Necessary for analyzing, making decisions, and suggesting next steps. A subscription key and region information are needed.
        """,
        unsafe_allow_html=True,
    )

def process_input(image_paths: List[str], 
                  tracking_id: str,
                  engagement_description: str, 
                  total_score: float, 
                  weighted_projected_acr: float, 
                  weighted_projected_length: float, 
                  weighted_partner_executives: float, 
                  weighted_actual_acr: float) -> Dict:
    """
    This function processes the input data and calculates various metrics.

    Args:
    image_paths (List[str]): A list of paths to the images related to the request.
    tracking_id (str): A unique identifier for the request.
    engagement_description (str): A string containing the description of the engagement.
    total_score (float): The total score calculated for the request.
    weighted_projected_acr (float): The weighted projected Annual Contract Revenue (ACR).
    weighted_projected_length (float): The weighted projected length of the project.
    weighted_partner_executives (float): The weighted number of partner executives involved in the project.
    weighted_actual_acr (float): The weighted actual Annual Contract Revenue (ACR).

    Returns:
    Dict: A dictionary containing the processed data and calculated metrics.
    """
    USER_PROMPT = f"""
        As an AI specialist in our organization, you are entrusted with the critical task of evaluating project requests through a detailed analysis process. This comprehensive process involves reviewing textual content, examining any supplementary visual materials, and considering quantitative inputs to calculate an evaluative score. Your role is to interpret these elements, aiming for strategic project enhancements while maintaining the confidentiality of our evaluation criteria.

        **Project Request Overview:**

        We receive submissions that address technological challenges necessitating advanced knowledge in AI and ML technologies:

        If submissions include images, these are assessed in subsequent stages to ensure a holistic evaluation.

        **Score Calculation and Assessment:**

        The evaluation score is derived using the `calculate_total_score` method, accounting for:
        - Projected Annual Contract Revenue (ACR): {weighted_projected_acr}
        - Projected Length of the Project: {weighted_projected_length}
        - Number of Partner Executives Involved: {weighted_partner_executives}
        - Actual ACR: {weighted_actual_acr}

        This method normalizes and weights the given inputs, calculating a total score that informs our evaluation. The precise weights used are proprietary.

        **Comprehensive Evaluation Procedure:**

        - **Request Analysis**: Start with an in-depth review of the submission's alignment with AI/ML technological demands, including GenAI applications. 

        - **Score and Inputs Interpretation**: Delve into the calculated score and its components, assessing the potential impact of each factor on the project's viability.

        - **Decision Process**: 
            - Consider project_description. Projects with clear, innovative, and technically detailed descriptions that effectively demonstrate the use of AI, ML, and GenAI technologies, particularly those proposing solutions for modern challenges, should be considered for approval.
            - For scores **below 0.1**, ALWAYS REJECT the project, providing detailed feedback on the reasons for rejection.
            - For scores **between 0.1 and 0.2**, projects must be carefully re-evaluated for their innovative potential and alignment with technological trends in AI and ML, providing targeted feedback for enhancement.
            - Scores **between 0.2 and 0.4** necessitate a focused review of the problem description to ascertain the project's eligibility for approval, emphasizing the project's strengths and developmental areas.
            - Scores **above 0.4** indicate a high likelihood of approval; however, a detailed assessment of the project's description and any visual data is essential to confirm its alignment with our technological standards and feasibility.

        - **Approval Criteria**: Projects must demonstrate a comprehensive and well-structured use of relevant AI/ML and GenAI technologies, supported by a detailed problem statement that aligns with current and future technological advancements in AI application development in the Consider project_description. 
             The calculated score should adhere to the rules specified in the "Decision Process" section. Specifically:
                - For scores **below 0.1**, ALWAYS REJECT the project, providing detailed feedback on the reasons for rejection.
                - For scores **between 0.1 and 0.2**, projects must demonstrate significant potential for innovation and alignment with technological trends in AI and ML.
                - For scores **between 0.2 and 0.4**, the problem description must provide sufficient detail and insight to justify approval, despite the middling score.
                - For scores **above 0.4**, the project description and any visual data must confirm the project's alignment with our technological standards and feasibility.

        - **Rejection Considerations**: Clearly articulate the reasons for any rejections, focusing on gaps in technological relevance or deficiencies in the project's conceptualization and proposed methodologies. Provide constructive feedback to guide future submissions towards alignment with AI, ML, and GenAI development goals. Specifically address the areas for improvement in the "Actionable Recommendations" section. 
            If the total score was the main influence for the rejection, provide detailed suggestions for improving the following components:
                - Projected Annual Contract Revenue (ACR): {weighted_projected_acr}
                - Projected Length of the Project: {weighted_projected_length}
                - Number of Partner Executives Involved: {weighted_partner_executives}
                - Actual ACR: {weighted_actual_acr}

        Calculated score: {total_score}
        project_description: {engagement_description}

        Take the necessary time to thoroughly evaluate the request. Consider all aspects of the submission and adhere to the guidelines and format provided above. 
         
        ### ❗Final Decision: 
        
        - ><Explicitly state the decision (**Approved** 👍 or **Rejected** 👎)>
        - ><Explicitly provide Tracking ID for monitoring purposes: {tracking_id}>

        ### 🤔 Evaluation Criteria:

        - **Request Analysis**:
        <Provide a detailed analysis of the request, focusing on both its content and context.>

        - **Score Interpretation**:
        <Discuss the calculated score, its implications, and any potential discrepancies between the numeric evaluation and qualitative assessment.>

        - **Decision Process**:
        <Explicitly state the decision (approve or reject), providing a reasoned argument that synthesizes the score analysis and request scrutiny.>

        - **Actionable Recommendations**:
        <Provide clear, structured guidance for next steps post-decision, tailored to the outcome (approved or rejected).>

        """

    
    # Call GPT4 Vision Manager
    response = st.session_state.gpt4_vision_manager.call_gpt4v_image(
        image_paths,
        system_instruction=settings.SYS_MESSAGE,
        user_instruction=USER_PROMPT,
        ocr=True,
        use_vision_api=True,
        display_image=False,
        max_tokens=2000,
        seed=42,
    )
    return response

with st.form("request_form", 
             clear_on_submit=False):
    bif_name = st.text_input("Request Title", "Request 1")
    bif_requestername = st.text_input("Requester", "John Doe")
    bif_requesteremail = st.text_input("Requester", "johnjoe@microsoft.com")
    bif_partner = st.multiselect("Partner", ["Yes", "No"], default=["Yes"])
    projected_work_hours = st.number_input("Projected Amount of Work (in hours)", value=10, min_value=1, max_value=60,
                                           help='''Please enter the estimated number of hours required to complete the work. 
                                           Be as specific as possible and try to be conservative in your estimate. The value should be between 1 and 60.''')

    expected_start_date = st.date_input("Expected Start Date", datetime.date.today() + datetime.timedelta(days=1),
                                        help='Please select the expected start date for the project. This date should be in the future.')
    # Check if the selected date is in the future
    if expected_start_date <= datetime.date.today():
        st.error('Error: The selected date is not in the future. Please select a future date.')    
    bif_msxid = st.text_input("MSXID", "5678")
    bif_topparentid = st.text_input("TPID", "1234")
    bif_primarysolutionareaname = st.multiselect("Primary Solution Area", ["AI/ML", "AI infra", "Knowledge Mining", "ML infra"], default=["AI/ML"])
    bif_secondarytechnologyname = st.multiselect("Secondary Solution Area", ["Application Development (LLMOps/MLOps)", "Enterprise AI/ML Architecture", "Infrastructure Optimization"], default=["Enterprise AI/ML Architecture"])
    bif_customername = st.text_input("Customer Name", "CustomerName Inc.")
    bif_ou = st.multiselect("Operating Unit", ["Retail/CPG", "FSI", "MFG", "HLS", "SDP", "West/Midwest", "South", "Northeast", "Southeast", "LATAM", "Canada", "Education", "Fed / Public Sector"])
    problem_description = st.text_area("Description of the Problem (make sure to add details to explain the problem)", 
                                       "The system is not responding.", help='''
        Craft a succinct business description that outlines the client's objectives, current utilization of Azure data and AI services, and the project timeline. 

        - **Objectives**: Detail the client's goals, such as enhancing efficiency or scalability, and specify Azure services like AOAI and/or AML. 
        - **Timeline**: Include a timeline for the completion of the engagement and highlight any existing challenges with the client's current infrastructure. Aim for clarity and brevity while effectively addressing the client's needs.

        **Example**: "Support the client in enhancing their legacy COBOL application infrastructure by developing a custom copilot tool utilizing GenAI capabilities for efficient code migration to Python. This initiative aims to improve operational efficiency, ensure scalability, and foster maintainability within the client's application ecosystem.

        **Azure Services Utilization**: The project will leverage Azure OpenAI (AOAI) for intelligent code translation and Azure Machine Learning (AML) to automate the migration process, optimizing the transition from COBOL to Python."''')
    projected_value = st.number_input("Projected ACR (in dollars)", value=10000, step=10000, 
                                      help='Please enter the projected Annual Contract Revenue (ACR) in dollars. The value must be between 0 (exclusive) and 50000 (inclusive).')

    if projected_value <= 0 or projected_value > 50000:
        st.error("The value must be between 0 (exclusive) and 50000 (inclusive).")

    necessary_skills = st.multiselect("Skills Necessary", ["Python", "Machine Learning", "Data Analysis", "Deep Learning", 
                                                           "Natural Language Processing", "Computer Vision", "Generative AI",
                                                             "RAG", "MLOps", "LLMLOps", "N/A"])    
    azure_ai_services = st.multiselect("Azure AI Services", ["AOAI", "Azure AI Search", "Azure Document Intelligence", 
                                                            "Azure Machine Learning", "Azure Bot Service", "Azure Databricks",
                                                            "Azure Anomaly Detector", "N/A"], default=["AOAI"])    
    engagement_country = st.text_input("Engagement Country", "USA", 
                                       help='Please enter the country where the engagement is taking place. For example, "USA".')
    engagement_region = st.text_input("Engagement Region", "Midwest",
                                      help='Please enter the region where the engagement is taking place. For example, "Midwest".')
    monthly_usage = st.number_input("Monthly Usage (in dollars)", value=10000, step=10000, 
                                help='Please enter the monthly ACR of the service in dollars. The value must be between 0 (exclusive) and 100000 (inclusive).')

    if monthly_usage <= 0 or monthly_usage > 100000:
        st.error("The value must be between 0 (exclusive) and 100000 (inclusive).")
    uploaded_file = st.file_uploader("Add an attachment")
    submit_button = st.form_submit_button("Submit")


# Main function to handle operations
def main():
    image_paths = []
    if submit_button:
        mandatory_fields = [bif_name, bif_requestername, bif_partner, projected_work_hours, expected_start_date, bif_msxid, bif_topparentid, 
                            bif_primarysolutionareaname, bif_secondarytechnologyname, bif_customername, problem_description, projected_value, 
                            bif_ou, azure_ai_services]

        field_names = ['Request Name', 'Requester', 'Partner', 'Projected Work Hours', 'Expected Start Date', 'MSXID', 'TPID', 
                       'Primary Solution Area', 'Secondary Solution Area', 'Customer Name', 'Problem Description', 'Projected ROI Value', 
                       'Operating Unit', 'Azure AI Services']

        if all(mandatory_fields):
            logger.info("Form submitted with values: %s", 
                        [bif_name, bif_requestername, bif_partner, projected_work_hours, expected_start_date, bif_msxid, bif_topparentid, 
                         bif_primarysolutionareaname, bif_secondarytechnologyname, bif_customername, problem_description, projected_value, 
                         bif_ou, azure_ai_services])
                
            # Add a file uploader
            if uploaded_file is not None:
                file_path = save_uploaded_file(uploaded_file)
                image_paths.append(file_path)

            # Get current time
            createdon = datetime.datetime.now().strftime("%Y-%m-%d")
            bif_requestid = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(8))

            with st.expander("Results"):
                # Process input if prompt is provided
                with st.spinner('Calculating score...'):
                    bif_parthner = 1 if "Yes" in bif_partner else 0
                    total_score, weighted_projected_acr, weighted_projected_length, weighted_partner_executives, weighted_actual_acr = calculate_total_score(
                        int(projected_value), int(projected_work_hours), bif_parthner, int(monthly_usage))
                    st.success(f"Qualification.ai Score successfully calculated: {total_score}")

                with st.spinner('Evaluating your request, please wait...'):
                    response = process_input(image_paths, bif_requestid, str(problem_description), total_score, weighted_projected_acr,
                                             weighted_projected_length, weighted_partner_executives, weighted_actual_acr)
                    st.write(response)

                # Send email to the AI GBB team aligned to the selected OUs
                selected_ous = set(bif_ou)
                mapped_ous = set(ou_email_mapping.keys())

                common_ous = selected_ous & mapped_ous

                if common_ous:
                    for ou in common_ous:
                        to_emails = ou_email_mapping[ou]
                        if to_emails:
                            send_email(response=response,
                                    from_email=FROM_EMAIL,
                                    to_emails=[to_emails],
                                    subject=f"Qualification.ai Request Evaluation - Tracking ID: {bif_requestid}")
                            st.success(f"Email sent successfully to the AI GBB team aligned to {ou}: {', '.join(to_emails)}")
                        else:
                            st.warning(f"No email addresses found for {ou}.")
                else:
                    st.error(f"None of the selected OUs found in email mapping.")

                 # Extract decision from response
                decision_match = re.search(r'\*\*(Approved|Rejected)\*\*', response)
                if decision_match:
                    decision = decision_match.group(1)
                    if decision == "Approved":
                        approved = "True"
                    elif decision == "Rejected": 
                        approved = "False"
                else:
                    st.error("Could not extract decision from response.")
                    decision = None

            request_data = {
                'RequestTitle': bif_name,
                'Requester': bif_requestername,
                'RequesterEmail': bif_requesteremail,
                'Partner': bif_partner[0] if bif_partner else None,
                'ProjectedWorkHours': projected_work_hours,
                'ExpectedStartDate': expected_start_date.strftime("%Y-%m-%d"),
                'MSXID': bif_msxid,
                'TPID': bif_topparentid,
                'PrimarySolutionArea': bif_primarysolutionareaname,
                'SecondarySolutionArea': bif_secondarytechnologyname,
                'CustomerName': bif_customername,
                'OperatingUnit': bif_ou,
                'ProblemDescription': problem_description,
                'ProjectedACR': projected_value,
                'NecessarySkills': necessary_skills,
                'AzureAIServices': azure_ai_services,
                'EngagementCountry': engagement_country,
                'EngagementRegion': engagement_region,
                'MonthlyUsage': monthly_usage,
                'Attachment': uploaded_file.read() if uploaded_file else None,
                'CreatedDate': createdon,
                'RequestId': bif_requestid,
                'Status': 'In progress' if approved == "True" else 'Blocked due to Rejection',
                'AssignedTo': to_emails, 
                'AssignedDate': createdon,
                'Approved': approved,
                'ApprovedDate': createdon,
                'ApprovedBy' : "QualiFiction.ai",
            }
                
            st.session_state.cosmos_client.index_data(data_list=[request_data], id_key="RequestId")

        else:
            logger.error("Form submitted but not all fields were filled.")
            for field, name in zip(mandatory_fields, field_names):
                if not field:
                    st.error(f"Please fill in the mandatory field: {name}")

      
# Call the main function
main()