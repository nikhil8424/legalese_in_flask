# LegalEase - Legal Document Processing Application

## Overview

LegalEase is a web-based application designed to process legal documents, extract key information, and simplify complex legal language. It leverages a combination of OCR (Optical Character Recognition), natural language processing (NLP), and machine learning (ML) techniques to provide a user-friendly experience.

**Key Features:**

*   **File Upload:** Supports uploading PDF and image files (.png, .jpg, .jpeg).
*   **Language Selection:** Allows users to select the language of the document (English, Marathi, Hindi, Tamil) for OCR.
*   **Text Extraction:** Extracts text from PDFs and images using Tesseract OCR and PyPDF2.
*   **Translation:** Translates extracted text into English using Google Translate.
*   **Document Type Identification:** Identifies the type of legal document (Contract, NDA, etc.) based on keywords.
*   **Section Extraction:** Extracts legal sections from the document.
*   **Key Point Extraction:** Extracts key information such as dates, organization names, city names, and involved individuals.
*   **Text Simplification:** Converts complex legal language into simpler English.
*   **Text Summarization:** Generates a concise summary of the document.
*  **Asynchronous Processing:** The backend processing of documents is handled using Celery to avoid blocking main web server.
*   **User-Friendly Interface:** Provides a clean and accessible interface through a web browser.

## Technologies Used

*   **Backend:**
    *   **Python:** Programming language.
    *   **Flask:** Web framework for creating the application.
    *   **Celery:** Asynchronous task queue for handling processing tasks in the background.
    *   **Redis:** In-memory data structure store used as Celery's broker and result backend.
    *   **PyPDF2:** Python library for extracting text from PDF documents.
    *   **pdf2image:** Python library for converting PDF pages to images
    *   **Pillow:** Python Imaging Library for image manipulation.
    *   **pytesseract:** Python wrapper for the Tesseract OCR engine.
    *   **deep-translator:** Python library for accessing translation services (Google Translate).
    *   **sumy:** Library for text summarization
    *   **spacy:** Library for NLP tasks, including name extraction.
    *   **nltk:** Natural language toolkit for text processing (tokenization).
    *   **transformers:** Hugging Face Transformers library for advanced NLP (paraphrasing).
    *   **textblob:** For simple text analysis.

*   **Frontend:**
    *   **HTML:**  Structure of the web pages.
    *   **CSS:** Styling of the web pages.
    *   **JavaScript** (Minimal):  For any additional client side functionalities.

## Setup and Installation

### Prerequisites

*   **Python:** Ensure you have Python 3.8 or higher installed. You can download Python from [https://www.python.org/downloads/](https://www.python.org/downloads/).
*   **Tesseract OCR:** Install the Tesseract OCR engine. Download from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki) (Choose appropriate installable).
*   **Redis:** Install Redis from [https://redis.io/download](https://redis.io/download). On Linux you can use `sudo apt install redis-server`.
*   **TESSDATA_PREFIX** Environment Variable: You may need to set your system environment variable to point to your tesseract training data folder location.

### Installation Steps

1.  **Clone the Repository:**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**

    *   **Windows:**
        ```bash
        venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Python Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
    Alternatively you can run the following `pip` commands.
    ```bash
    pip install Flask
    pip install celery
    pip install redis
    pip install PyPDF2
    pip install pdf2image
    pip install Pillow
    pip install pytesseract
    pip install deep-translator
    pip install sumy
    pip install spacy
    pip install nltk
    pip install transformers
    pip install textblob
    ```
    
    Then install spacy English model.
    ```bash
        python -m spacy download en_core_web_sm
    ```

5.  **Environment Variables** Set the `TESSDATA_PREFIX` to your tesseract training data path in your system environment variables.

6.  **Configure Tesseract Path** Verify the `pytesseract.pytesseract.tesseract_cmd` in `app.py` and change the executable path as needed. The path is currently set to `C:\Program Files\Tesseract-OCR\tesseract.exe`.

7.  **Run the Application:**

    ```bash
    python app.py
    ```

8.  **Access the Application:**

    *   Open a web browser and go to `http://127.0.0.1:5000/`

## How to Use

1.  **Upload Document:** On the home page, select the document's language using the dropdown menu and upload your PDF or image file using the file upload input.
2.  **Process Document:** Once uploaded, the document will be processed, and you will be redirected to the processing page.
3.  **Analyze:** Once processing is complete, you can perform various analysis on the document using the available buttons.
   * Click the respective buttons `Identify Document`, `Key Points`, `Simplify Text` and `Generate Summary` to view those results.
4.  **View Results:** The results will be displayed on the `result.html` page.

## Folder Structure

your_project/
├── app.py # Main Flask application file
├── requirements.txt # Python dependencies
├── templates/ # HTML templates
│ ├── index.html # Home page for document upload
│ ├── process.html # Page that displays processed text and analysis buttons
│ └── result.html # Page for displaying analysis results
├── uploads/ # Folder where uploaded files are stored
└── venv/ # Virtual environment (created when setup)

## Contributing

If you would like to contribute to this project, feel free to create a pull request.

## License

[Choose a license like MIT, Apache 2.0, etc.]

## Disclaimer

This application is intended for informational purposes only and should not be used for any critical legal decisions without consulting an attorney.
