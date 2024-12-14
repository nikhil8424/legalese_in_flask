from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
import os
import logging
import time
from celery import Celery
import PyPDF2
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from deep_translator import GoogleTranslator
from collections import defaultdict
import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer
import spacy
import nltk
from nltk.corpus import wordnet
from transformers import pipeline
from collections import Counter
from textblob import TextBlob

nltk.download('punkt')
nltk.download('punkt_tab')

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
CELERY_BROKER_URL = 'redis://localhost:6379/0'  # Redis broker URL
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

# --- Logging ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Flask App ---
app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Celery Setup ---
celery = Celery(app.name, broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
celery.conf.update(app.config)

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Pre-compile regular expression patterns for efficiency
SECTION_PATTERN = re.compile(r'\bSection\s+\d+\b', re.IGNORECASE)
DATE_PATTERNS = [
    re.compile(r"(?:January|February|March|April|May|June|July|August|September|October|November|December)[\s\n]*\d{1,2},[\s\n]*\d{4}"),
    re.compile(r"\d{1,2}[\s\n]*(?:January|February|March|April|May|June|July|August|September|October|November|December)[,\s\n]*\d{4}"),
    re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}"),
    re.compile(r"\d{1,2}[-/]\d{2}[-/]\d{4}"),
    re.compile(r"\d{2}[-/]\d{2}[-/]\d{4}"),
    re.compile(r"\d{1,2}[\s\n]*(?:January|February|March|April|May|June|July|August|September|October|November|December)[\s\n]*\d{4}"),
]
ORGANIZATION_SUFFIXES = [
    r"Pvt\.? Ltd\.?", r"Ltd\.?", r"LLP", r"LP", r"PA", r"PC", r"NPO", r"NGO", r"Foundation", r"Properties",
    r"Co-op", r"Cooperative Society", r"Trust", r"Section 8 Company", r"Inc\.?", r"Corp\.?", r"LLC", r"PLC",
    r"GmbH", r"S\.A\.", r"S\.R\.L\.", r"A\.G\.", r"KGaA"
]
ORGANIZATION_PATTERN = re.compile(r"\b[A-Z][A-Za-z\s&'-]*?\b(?:\s(?:'|\b[A-Z][A-Za-z\s&'-]+?\b))*(?:" + "|".join(ORGANIZATION_SUFFIXES) + r")\b")

# Load spacy model for name extraction, loading outside of function
NLP = spacy.load("en_core_web_sm")


# --- Celery Task ---
@celery.task(bind=True)
def process_document_task(self, filepath, language_code):
    """
    Celery task for processing the document.
    """
    try:
        logging.debug(f"Starting document processing for {filepath}, Language: {language_code}")
        start_time = time.time()

        extracted_text = extract_text(filepath, language_code)
        translated_text = translate_to_english(extracted_text)

        end_time = time.time()
        logging.debug(f"Document processed in {end_time - start_time} seconds.")

        return translated_text
    except Exception as e:
        logging.error(f"Error in processing document: {e}")
        return None

# --- Routing ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        #handle file uploads and process data
        if 'file' not in request.files:
             return "No file part in request"
        file = request.files['file']
        if file.filename == '':
           return "No file selected"
        if file and allowed_file(file.filename):
             filename = secure_filename(file.filename)
             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
             file.save(filepath)
             session['filepath'] = filepath
             session['language'] = request.form.get("language")
             
             language_code = language_dict.get(session['language'])
             task = process_document_task.delay(filepath, language_code)
             session['task_id'] = task.id
             return redirect(url_for('processing_page'))
             
    return render_template("index.html", selected_language=session.get('language', 'English'))

@app.route('/processing')
def processing_page():
    return render_template("processing.html")

@app.route("/process", methods=["GET"])
def process_document():
    task_id = session.get('task_id')

    if not task_id:
        return "Error: Task ID not found in session"

    task = celery.AsyncResult(task_id)
    if task.state == "PENDING":
        return render_template("processing.html") #Show a loading page
    elif task.state != "SUCCESS":
        return "Error: Processing failed or timed out."
    
    translated_text = task.result
    session['extracted_text'] = translated_text
    selected_language = session.get('language')
    return render_template("process.html", translated_text = session.get('extracted_text'), selected_language = selected_language)


# Function to convert legal language into simple english language
class TextSimplifier:
    def __init__(self):
        self.pos_map = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r'}

        try:
            self.paraphraser = pipeline("text2text-generation", model="t5-small")
            logging.info("Model initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing model: {e}")

    def find_simple_synonym(self, word, pos_tag):
        synonyms = wordnet.synsets(word)
        if synonyms:
            filtered_synonyms = [lemma.name().replace('_', ' ') for syn in synonyms 
                                for lemma in syn.lemmas() if syn.pos() == self.pos_map.get(pos_tag, 'n')]
            if filtered_synonyms:
                common_synonym = Counter(filtered_synonyms).most_common(1)[0][0]
                return common_synonym
        return word

    def paraphrase_sentence(self, sentence):
        try:
            paraphrased = self.paraphraser(sentence, max_length=100, num_return_sequences=1)
            return paraphrased[0]['generated_text']
        except Exception as e:
            logging.error(f"Paraphrasing error: {e}")
            return sentence

    def simplify_sentence(self, sentence):
        simplified_sentence = []
        for token in sentence:
            pos_tag = token.pos_
            if pos_tag in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                simplified_word = self.find_simple_synonym(token.text, pos_tag)
                simplified_sentence.append(simplified_word)
            else:
                simplified_sentence.append(token.text)
        return ' '.join(simplified_sentence)

    def simplify_text(self, text):
        doc = NLP(text) #use loaded spacy model
        simplified_sentences = []

        for sentence in doc.sents:
            simplified_sentence = self.simplify_sentence(sentence)
            simplified_text = self.paraphrase_sentence(simplified_sentence)
            simplified_sentences.append(simplified_text)

        return ' '.join(simplified_sentences)

# Function to summarize text using Sumy
def summarize_text(text):
    start_time = time.time()
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  # Adjust the number of sentences in the summary
    summary_text = "\n".join(str(sentence) for sentence in summary)
    end_time = time.time()
    logging.debug(f"Text summarized in {end_time - start_time} seconds")
    return summary_text

# Function to extract text from image using pytesseract
def extract_text_from_image(image_path, language):
    start_time = time.time()
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang=language)
    end_time = time.time()
    logging.debug(f"Text extracted from image in {end_time - start_time} seconds")
    return text

# Function to extract text from PDF using PyPDF2
def pdf_to_text(pdf_path):
    start_time = time.time()
    text = ''
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ''  # Handle None if text extraction fails
    end_time = time.time()
    logging.debug(f"Text extracted from PDF using PyPDF2 in {end_time - start_time} seconds")
    return text

# Function to convert PDF to images and then extract text
def pdf_to_images_and_text(pdf_path, language):
    start_time = time.time()
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    images = convert_from_path(pdf_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image, lang=language)
    end_time = time.time()
    logging.debug(f"Text extracted from PDF using images in {end_time - start_time} seconds")
    return text

# Function to handle file upload and text extraction
def extract_text(file_path, language_code):
    start_time = time.time()
    if file_path.lower().endswith('.pdf'):
        if language_code == "eng":
            extracted_text = pdf_to_text(file_path)
        else:
            extracted_text = pdf_to_images_and_text(file_path, language_code)
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        extracted_text = extract_text_from_image(file_path, language_code)
    else:
        raise ValueError("Only PDF and image files are supported.")
    end_time = time.time()
    logging.debug(f"Text extracted from file in {end_time - start_time} seconds")
    return extracted_text

# Function to translate extracted text to English
def translate_to_english(text):
    start_time = time.time()
    try:
        translator = GoogleTranslator(target='en')
        lines = text.split('\n')
        batch_size = 100
        translated_text = ""
        for i in range(0, len(lines), batch_size):
            batch = lines[i:i + batch_size]
            batch_text = "\n".join(batch)
            translated_batch = translator.translate(batch_text)
            translated_text += translated_batch + "\n"
        end_time = time.time()
        logging.debug(f"Text translated in {end_time - start_time} seconds")
        return translated_text.strip()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return f"An error occurred: {e}"

# Function to extract laws and sections from the text
def extract_laws_and_sections(text):
    return SECTION_PATTERN.findall(text)

# Function to identify document type and extract sections
def identify_document_type(text):
    text = text.lower()
    keyword_counts = defaultdict(int)
    for doc_type, keywords in legal_keywords.items():
        for keyword in keywords:
            if keyword.lower() in text:
                keyword_counts[doc_type] += 1
    
    sections = extract_laws_and_sections(text)
    
    if keyword_counts:
        max_type = max(keyword_counts, key=keyword_counts.get)
        return max_type, sections, keyword_counts
    else:
        return 'Unknown Document Type', sections, keyword_counts

# Legal keywords dictionary
legal_keywords = {
    'Contract': ['Payment Terms', 'Confidentiality', 'Agreement', 'Breach', 'Term', 'Consideration', 'Covenant', 'Obligations', 'Execution', 'Termination'],
    'Non-Disclosure Agreement': ['Confidentiality', 'Non-Disclosure', 'Proprietary Information', 'Trade Secrets', 'Disclosure', 'Recipient', 'Restricted', 'Confidential Information'],
    'Lease Agreement': ['Lease Term', 'Rent', 'Landlord', 'Tenant', 'Deposit', 'Premises', 'Maintenance', 'Utilities', 'Renewal', 'Termination'],
    'Employment Contract': ['Employment', 'Salary', 'Benefits', 'Job Title', 'Termination', 'Probation', 'Non-Compete', 'Work Hours', 'Duties', 'Confidentiality'],
    'Partnership Agreement': ['Partnership', 'Profit Sharing', 'Responsibilities', 'Contributions', 'Duties', 'Liabilities', 'Term', 'Dispute Resolution', 'Dissolution'],
    'Loan Agreement': ['Loan Amount', 'Interest Rate', 'Repayment Terms', 'Collateral', 'Default', 'Lender', 'Borrower', 'Principal', 'Amortization', 'Term'],
    'Service Agreement': ['Services', 'Deliverables', 'Scope of Work', 'Compensation', 'Performance', 'Terms', 'Termination', 'Liability', 'Indemnification', 'Warranties'],
    'Settlement Agreement': ['Settlement', 'Claims', 'Dispute', 'Compensation', 'Release', 'Terms', 'Agreement', 'Confidentiality', 'Payment', 'Resolution'],
    'Privacy Policy': ['Personal Data', 'Privacy', 'Data Collection', 'Usage', 'Disclosure', 'Security', 'Rights', 'Cookies', 'Third Parties', 'Compliance'],
    'Power of Attorney': ['Authority', 'Agent', 'Principal', 'Powers', 'Representation', 'Decision-Making', 'Legal', 'Termination', 'Revocation', 'Duties'],
    'Criminal Case': ['Defendant', 'Plaintiff', 'Charges', 'Evidence', 'Trial', 'Verdict', 'Prosecution', 'Defense', 'Sentencing', 'Appeal'],
    'Civil Case': ['Plaintiff', 'Defendant', 'Complaint', 'Evidence', 'Trial', 'Judgment', 'Settlement', 'Damages', 'Appeal', 'Liability'],
    'Divorce Case': ['Petitioner', 'Respondent', 'Custody', 'Alimony', 'Division of Assets', 'Settlement', 'Support', 'Visitation', 'Child Support', 'Dissolution'],
    'Property Case': ['Plaintiff', 'Defendant', 'Property Title', 'Lease', 'Possession', 'Easement', 'Boundary', 'Transfer', 'Sale', 'Ownership'],
    'Intellectual Property': ['Patent', 'Trademark', 'Copyright', 'Infringement', 'Licensing', 'Royalty', 'Patent Application', 'Trademark Registration', 'Creative Works', 'Protection'],
    'Bankruptcy': ['Debtor', 'Creditor', 'Filing', 'Discharge', 'Repayment Plan', 'Trustee', 'Chapter 7', 'Chapter 11', 'Chapter 13', 'Liquidation'],
    'Family Law': ['Custody', 'Support', 'Adoption', 'Guardianship', 'Marriage', 'Divorce', 'Paternity', 'Child Welfare', 'Protective Orders', 'Family Mediation'],
    'Immigration': ['Visa', 'Residency', 'Citizenship', 'Deportation', 'Green Card', 'Asylum', 'Work Permit', 'Naturalization', 'Immigration Status', 'Application'],
    'Contract Dispute': ['Breach', 'Enforcement', 'Remedies', 'Damages', 'Settlement', 'Negotiation', 'Contract Terms', 'Performance', 'Dispute Resolution', 'Mediation'],
    'Insurance Claim': ['Policy', 'Claim', 'Coverage', 'Exclusions', 'Premiums', 'Deductibles', 'Settlement', 'Claim Denial', 'Evidence', 'Payout'],
    'Traffic Violation': ['Citation', 'Ticket', 'Fine', 'Court Appearance', 'Violation', 'Penalty', 'Evidence', 'Defendant', 'Judge', 'Appeal'],
}

def extract_dates(text):
    dates_found = set()
    for pattern in DATE_PATTERNS:
        dates_found.update(match.group() for match in pattern.finditer(text))
    return list(dates_found)

def extract_organization_names(text):
    cleaned_text = clean_text(text)
    organizations = set()
    for match in ORGANIZATION_PATTERN.finditer(cleaned_text):
         org_name = match.group().strip()
         if len(org_name.split()) <= 6 and not re.search(r'\b(?:dispute|arising|relating|agreement|binding|arbitration|rules|association|remainder|document|omitted|brevity|terms|ownership|website|code|warranty|defects|limitations|liability|witness|whereof|parties|executed|date|first|written|above|inc)\b', org_name, re.IGNORECASE):
             organizations.add(org_name)
    return list(organizations)


def extract_city_names(text, MAHARASHTRA_CITIES):
    city_pattern = r'\b(?:' + '|'.join(re.escape(city) for city in MAHARASHTRA_CITIES) + r')\b'
    cities_found = set()
    cleaned_text = clean_text(text)
    for match in re.finditer(city_pattern, cleaned_text, re.IGNORECASE):
        city_name = match.group().strip()
        if city_name in MAHARASHTRA_CITIES and len(city_name) > 1:
            cities_found.add(city_name)
    return list(cities_found)

def extract_names(text):
    doc = NLP(text) #use loaded spacy model
    filtered_names = set()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            normalized_name = ' '.join(name.split())
            if len(normalized_name.split()) > 1 and not re.match(r'\b(?:This|Witness|Whereof|Sealed|Witnesseth|Principal|a|b|c|d)\b', normalized_name):
                filtered_names.add(normalized_name)
    filtered_names = list(filtered_names)
    filtered_names = [name for name in filtered_names if re.match(r'^[A-Za-z\s]+$', name)]
    return filtered_names

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text
    
    
MAHARASHTRA_CITIES = [
    "Visakhapatnam", "Vijayawada", "Guntur", "Tirupati", "Kakinada", "Rajahmundry", "Nellore", "Delhi",
            "Anantapur", "Kadapa", "Chittoor", "Eluru", "Srikakulam", "Prakasam", "Kurnool", 
            "Vijayanagaram", "Bhimavaram", "Machilipatnam", "Palnadu", "Narsipatnam", "Tanuku", 
            "Yemmiganur", "Tadipatri", "Jammalamadugu", "Peddaganjam", "Araku Valley",
            "Itanagar", "Tawang", "Bomdila", "Ziro", "Naharlagun", "Pasighat", "Roing", "Changlang", 
            "Tezu", "Daporijo",
            "Guwahati", "Dibrugarh", "Silchar", "Jorhat", "Nagaon", "Tezpur", "Haflong", "Karimganj", 
            "Barpeta", "Dhemaji", "Bongaigaon", "Sonitpur", "Sivasagar", "Golaghat", "Cachar", 
            "Kokrajhar", "Lakhimpur", "Nalbari", "Darrang", "Kamrup", "Dhubri",
            "Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Darbhanga", "Munger", "Purnia", "Arrah", 
            "Siwan", "Sasaram", "Nalanda", "Katihar", "Begusarai", "Jehanabad", "Samastipur", "Chhapra", 
            "Buxar", "Kishanganj", "Madhubani", "Saran", "Gopalganj", "Bettiah",
            "Raipur", "Bilaspur", "Korba", "Durg", "Jagdalpur", "Raigarh", "Ambikapur", "Rajnandgaon", 
            "Janjgir-Champa", "Kanker", "Dhamtari", "Bemetara", "Kabirdham", "Baloda Bazar", "Gariaband", 
            "Surguja", "Mungeli", "Sukma", "Bijapur", "Balrampur",
            "Panaji", "Margao", "Vasco da Gama", "Mapusa", "Ponda", "Quepem", "Cortalim", "Bicholim", 
            "Sanquelim", "Pernem", "Canacona", "Sanguem", "Kankon",
            "Ahmedabad", "Surat", "Vadodara", "Rajkot", "Junagadh", "Bhavnagar", "Gandhinagar", 
            "Anand", "Nadiad", "Mehsana", "Vapi", "Navsari", "Amreli", "Porbandar", "Kutch", "Patan", 
            "Jamnagar", "Bharuch", "Valsad", "Gir Somnath", "Dahod", "Surendranagar", "Aravalli", 
            "Kheda", "Mahesana", "Sabarkantha", "Palitana", "Modasa", "Narmada",
            "Faridabad", "Gurgaon", "Hisar", "Rohtak", "Ambala", "Karnal", "Panipat", "Yamunanagar", 
            "Jind", "Sirsa", "Kurukshetra", "Panchkula", "Mahendragarh", "Bhiwani", "Fatehabad", 
            "Rewari", "Nuh", "Palwal", "Kaithal", "Pehowa", "Tosham", "Hansi",
            "Shimla", "Manali", "Dharamshala", "Kullu", "Solan", "Hamirpur", "Mandi", "Kangra", 
            "Bilaspur", "Una", "Nahan", "Chamba", "Palampur", "Sundernagar",
            "Srinagar", "Jammu", "Udhampur", "Rajouri", "Kathua", "Anantnag", "Pulwama", "Baramulla", 
            "Kargil", "Leh", "Samba", "Doda", "Reasi", "Poonch", "Kupwara", "Bandipora",
            "Ranchi", "Jamshedpur", "Dhanbad", "Hazaribagh", "Giridih", "Bokaro", "Deoghar", 
            "Chaibasa", "Medininagar", "Koderma", "Daltonganj", "Jhumri Telaiya", "Raghubar Nagar", 
            "Ramgarh", "Jamtara", "Pakur", "Godda", "Sahebganj", "Dumka", "Palamu", "Giridih", 
            "Hazaribagh",
            "Bengaluru", "Mysuru", "Hubli", "Dharwad", "Mangalore", "Belgaum", "Shimoga", "Tumkur", 
            "Chitradurga", "Kolar", "Udupi", "Hassan", "Bagalkot", "Bijapur", "Bidar", "Raichur", 
            "Bellary", "Gulbarga", "Devadurga", "Yadgir", "Chikkamagalur", "Chikkaballapur", 
            "Mandya", "Haveri", "Kodagu", "Karwar", "Davangere", "Puttur", "Sagara", "Bantwal", 
            "Koppal", "Hospet", "Gadag", "Haveri", "Srinivaspur",
            "Thiruvananthapuram", "Kochi", "Kozhikode", "Kannur", "Palakkad", "Alappuzha", 
            "Malappuram", "Kottayam", "Idukki", "Ernakulam", "Wayanad", "Pathanamthitta", 
            "Kasaragod", "Ponnani", "Perinthalmanna", "Thrissur", "Changanassery", "Muvattupuzha", 
            "Attingal", "Varkala", "Sreekariyam",
            "Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain", "Sagar", "Satna", "Rewa", 
            "Ratlam", "Burhanpur", "Dewas", "Khargone", "Hoshangabad", "Shivpuri", "Mandsaur", 
            "Chhindwara", "Datiya", "Tikamgarh", "Ujjain", "Sehore", "Betul", "Jhabua", "Agar Malwa", 
            "Shahdol", "Anuppur", "Pachmarhi", "Khandwa",
            "Mumbai", "Pune", "Nagpur", "Aurangabad", "Nashik", "Thane", "Kolhapur", "Solapur", 
            "Amravati", "Jalgaon", "Latur", "Satara", "Ratnagiri", "Sindhudurg", "Raigad", "Palghar", 
            "Wardha", "Buldhana", "Akola", "Washim", "Yavatmal", "Osmanabad", "Jalna", "Parbhani", 
            "Beed", "Nanded", "Hingoli", "Dhule", "Jalgaon", "Nashik",
            "Imphal", "Shillong", "Aizawl", "Kohima", "Agartala", "Itanagar", "Dimapur", 
            "Lunglei", "Tura", "Churachandpur", "Silchar", "Kohima", "Mon", "Wokha",
            "Guwahati", "Silchar", "Dibrugarh", "Tezpur", "Jorhat", "Nagaon", "Karimganj", 
            "Hailakandi", "Kokrajhar", "Bongaigaon", "Dhemaji", "Golaghat", "Sivasagar", 
            "Barpeta", "Dhubri", "Nalbari", "Kamrup", "Kokrajhar", "Sonitpur", "Lakhimpur", 
            "Cachar", "Darrang", "Mikir Hills",
            "Lucknow", "Kanpur", "Agra", "Varanasi", "Allahabad", "Meerut", "Ghaziabad", "Noida", 
            "Aligarh", "Bareilly", "Mathura", "Moradabad", "Saharanpur", "Faizabad", "Ambedkar Nagar", 
            "Sultanpur", "Rae Bareli", "Etawah", "Mainpuri", "Unnao", "Mau", "Basti", "Jaunpur", 
            "Deoria", "Gorakhpur", "Firozabad", "Shahjahanpur", "Rampur", "Azamgarh", "Ballia", 
            "Bahraich", "Sitapur", "Hardoi", "Lakhimpur Kheri", "Gonda", "Siddharth Nagar", "Pilibhit",
            "Dehradun", "Haridwar", "Nainital", "Mussoorie", "Roorkee", "Haldwani", "Rudrapur", 
            "Kashipur", "Rishikesh", "Pauri", "Almora", "Bageshwar", "Champawat", "Uttarkashi", 
            "Tehri", "Pithoragarh", "Udham Singh Nagar",
            "Kolkata", "Howrah", "Darjeeling", "Siliguri", "Asansol", "Durgapur", "Kalyani", 
            "Bankura", "Murarai", "Haldia", "Malda", "Bardhaman", "Jamshedpur", "Durgapur", 
            "Purulia", "Krishnanagar", "Bongaigaon", "Jalpaiguri", "Midnapore", "Cooch Behar", 
            "Alipurduar"
]
# Language dictionary for mapping language names to Tesseract codes
language_dict = {
    "Marathi": "mar",
    "English": "eng",
    "Hindi": "hin",
    "Tamil": "tam"
}

@app.route('/simplify', methods=["GET"])
def simplify_text_route():
     if not session.get('extracted_text'):
           return "No text extracted yet."
     simplified_text =  TextSimplifier().simplify_text(session.get('extracted_text'))
     return render_template("result.html", analysis_result = simplified_text)

@app.route('/identify_doc', methods=["GET"])
def identify_document_route():
      if not session.get('extracted_text'):
           return "No text extracted yet."
      doc_type, sections, keyword_counts = identify_document_type(session.get('extracted_text'))
      sections_text = "\n".join(sections) if sections else "No sections found."
        
        # Create a detailed keyword count message
      keyword_count_details = "\n".join([f"{keyword}: {count}" for keyword, count in keyword_counts.items()])
      result_message = (f"The document type is: {doc_type}\n\n"
                      f"Sections Found:\n{sections_text}\n\n"
                      f"Keyword Counts:\n{keyword_count_details}")
      
      return render_template("result.html", analysis_result = result_message)

@app.route('/key_points', methods=["GET"])
def key_points_route():
        if not session.get('extracted_text'):
             return "No text extracted yet."

        extracted_text = session.get('extracted_text')
        
        dates = extract_dates(extracted_text)
        organizations = extract_organization_names(extracted_text)
        city_names = extract_city_names(extracted_text, MAHARASHTRA_CITIES)
        names = extract_names(extracted_text)

        # Create detailed messages for each extracted item
        dates_text = "\n".join([f"{i}. Date: {date}" for i, date in enumerate(dates, start=1)]) if dates else "No dates found."
        organizations_text = "\n".join([f"{i}. Organization: {org}" for i, org in enumerate(organizations, start=1)]) if organizations else "No organization names found."
        cities_text = "\n".join([f"{i}. Location: {city}" for i, city in enumerate(city_names, start=1)]) if city_names else "No city names found."
        names_text = "\n".join([f"{i}. Name: {name}" for i, name in enumerate(names, start=1)]) if names else "No human names found."

        # Combine all results into a single message
        result_message = (f"Dates Extracted:\n{dates_text}\n\n"
                        f"Organization Names Extracted:\n{organizations_text}\n\n"
                        f"City Names Extracted:\n{cities_text}\n\n"
                        f"People Involved:\n{names_text}")

        # Show the result in a message box
        return render_template("result.html", analysis_result = result_message)

@app.route('/summary', methods=["GET"])
def generate_summary_route():
        if not session.get('extracted_text'):
            return "No text extracted yet."
        
        summary = summarize_text(session.get('extracted_text'))
        if summary:
           return render_template("result.html", analysis_result = summary)
        else:
            return "No summary available."
    
if __name__ == "__main__":
    app.run(debug=True)