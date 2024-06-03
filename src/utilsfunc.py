import os
import uuid
from typing import Any, Dict
from datetime import datetime
import requests
import time
import json
from utils.ml_logging import get_logger

# Initialize logging
logger = get_logger()

def save_uploaded_file(uploaded_file) -> str:
    """
    Save an uploaded file to a specified directory.

    :param uploaded_file: Streamlit UploadedFile object.
    :return: Path to the saved file.
    """
    upload_directory = os.path.join("utils", "uploads")
    if not os.path.exists(upload_directory):
        os.makedirs(upload_directory)

    conversation_id = str(uuid.uuid4())
    date_str = datetime.now().strftime("%Y%m%d")

    subdirectory = os.path.join(upload_directory, f"{conversation_id}_{date_str}")
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    file_path = os.path.join(subdirectory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def clean_json_string(json_string: str) -> str:
    """
    Cleans a JSON string by removing unwanted characters.

    :param json_string: The JSON string to be cleaned.
    :return: The cleaned JSON string.
    """
    cleaned_string = (
        json_string.replace("```json\n", "").replace("\n", "").replace("```", "")
    )
    return cleaned_string


def compare_invoices(
    invoice_document_intelligence: Dict[str, Any],
    invoice_gpt4_vision: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Compares key invoice details from two sources against the ground truth.

    :param invoice_document_intelligence: The invoice details extracted by Document Intelligence.
    :param invoice_gpt4_vision: The invoice details extracted by GPT-4 Vision.
    :param ground_truth: The actual invoice details.
    :return: A dictionary with the comparison results.
    """

    keys_to_compare = ["TotalTax", "InvoiceTotal", "InvoiceId", "InvoiceDate"]

    return {
        key: {
            "invoice_document_intelligence": invoice_document_intelligence.get(
                key, {}
            ).get("content"),
            "invoice_gpt4_vision": invoice_gpt4_vision.get(key, {}).get("content"),
            "ground_truth": ground_truth.get(key),
        }
        for key in keys_to_compare
    }


