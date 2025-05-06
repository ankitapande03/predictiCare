import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import base64
from io import BytesIO
from PIL import Image
import torch
from facenet_pytorch import MTCNN
import json
import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Multimodal Disease Detection",
    page_icon="🔬",
    layout="wide"
)

# Application title
st.title("Multimodal Disease Detection: Facial Biomarker Analysis")

# Initialize session state
if 'biomarker_results' not in st.session_state:
    st.session_state.biomarker_results = None
if 'diagnosis' not in st.session_state:
    st.session_state.diagnosis = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None

# Load face detection model
@st.cache_resource
def load_face_detector():
    return MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

face_detector = load_face_detector()

# Groq client initialization
@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = st.secrets.get("GROQ_API_KEY", "")
    
    if not api_key:
        st.warning("GROQ_API_KEY not found. Please set it in your .env file or Streamlit secrets.")
        return None
    
    return groq.Client(api_key=api_key)

# Biomarker extraction function
def extract_facial_biomarkers(image):
    """Extract facial biomarkers from an image."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    boxes, probs = face_detector.detect(img_tensor)
    
    if boxes is None:
        return None, "No face detected in the image."
    
    box = boxes[0].tolist()
    x1, y1, x2, y2 = [int(coord) for coord in box]
    face = image[y1:y2, x1:x2]
    face_hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
    
    biomarkers = {}
    biomarkers['skin_hue'] = float(np.mean(face_hsv[:, :, 0]))
    biomarkers['skin_saturation'] = float(np.mean(face_hsv[:, :, 1]))
    biomarkers['skin_value'] = float(np.mean(face_hsv[:, :, 2]))
    
    height, width = face.shape[:2]
    left_half = face[:, :width//2]
    right_half = face[:, width//2:]
    right_half_flipped = cv2.flip(right_half, 1)
    
    if left_half.shape != right_half_flipped.shape:
        min_height = min(left_half.shape[0], right_half_flipped.shape[0])
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half = left_half[:min_height, :min_width]
        right_half_flipped = right_half_flipped[:min_height, :min_width]
    
    symmetry_diff = np.mean(np.abs(left_half.astype(np.float32) - right_half_flipped.astype(np.float32)))
    symmetry_score = 1 - (symmetry_diff / 255)
    biomarkers['face_symmetry'] = float(symmetry_score)
    
    eye_region = face[int(height*0.2):int(height*0.5), :]
    eye_hsv = cv2.cvtColor(eye_region, cv2.COLOR_RGB2HSV)
    red_mask = cv2.inRange(eye_hsv, (0, 100, 100), (10, 255, 255))
    red_percentage = np.sum(red_mask > 0) / (eye_region.shape[0] * eye_region.shape[1])
    biomarkers['eye_redness'] = float(red_percentage)
    
    yellow_mask = cv2.inRange(face_hsv, (20, 100, 100), (30, 255, 255))
    yellow_percentage = np.sum(yellow_mask > 0) / (face.shape[0] * face.shape[1])
    biomarkers['yellowing'] = float(yellow_percentage)
    
    pallor_mask = cv2.inRange(face_hsv, (0, 0, 200), (180, 50, 255))
    pallor_percentage = np.sum(pallor_mask > 0) / (face.shape[0] * face.shape[1])
    biomarkers['pallor'] = float(pallor_percentage)
    
    return biomarkers, face

# RAG pipeline setup
@st.cache_resource
def initialize_rag():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists("faiss_index"):
        vector_store = FAISS.load_local("faiss_index", embeddings)
        return vector_store
    
    sample_texts = [
        "Facial pallor is often associated with anemia, which can be caused by iron deficiency, vitamin B12 deficiency, or chronic diseases.",
        "Yellowing of the skin and eyes (jaundice) is a common sign of liver problems, such as hepatitis or cirrhosis.",
        "Facial asymmetry can be a sign of Bell's palsy, stroke, or other neurological conditions.",
        "Eye redness can indicate conjunctivitis, allergies, or more serious conditions like glaucoma.",
        "Cyanosis, a bluish discoloration of the skin, suggests poor blood oxygenation and could indicate heart or lung disease.",
        "Skin texture changes, such as roughness or rashes, can be signs of various dermatological conditions.",
        "Facial edema (swelling) might indicate kidney problems, allergic reactions, or thyroid disorders.",
        "Excessive sweating visible on the face may be a sign of hyperthyroidism or infection.",
        "Butterfly rash across the cheeks and nose is a classic sign of lupus.",
        "Drooping eyelids can indicate myasthenia gravis or other neuromuscular disorders."
    ]
    
    documents = [{"content": text} for text in sample_texts]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([doc["content"] for doc in documents])
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local("faiss_index")
    
    return vector_store

def retrieve_medical_info(biomarkers):
    vector_store = initialize_rag()
    query = "Medical conditions associated with "
    
    if biomarkers['pallor'] > 0.5:
        query += "facial pallor, "
    if biomarkers['yellowing'] > 0.1:
        query += "yellowing of skin (jaundice), "
    if biomarkers['face_symmetry'] < 0.8:
        query += "facial asymmetry, "
    if biomarkers['eye_redness'] > 0.1:
        query += "eye redness, "
    
    if query == "Medical conditions associated with ":
        query = "Common medical conditions detectable through facial analysis"
    
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

def query_llm_with_groq(biomarkers, context):
    client = get_groq_client()
    
    if client is None:
        st.error("Groq client initialization failed. Please check your API key.")
        return None
    
    prompt = f"""
    You are a medical AI assistant analyzing facial biomarkers to detect potential health conditions.
    
    Biomarkers detected:
    - Skin hue: {biomarkers['skin_hue']:.2f}
    - Skin saturation: {biomarkers['skin_saturation']:.2f}
    - Skin value (brightness): {biomarkers['skin_value']:.2f}
    - Face symmetry score: {biomarkers['face_symmetry']:.2f} (higher is more symmetrical)
    - Eye redness: {biomarkers['eye_redness']:.4f} (percentage of red pixels in eye region)
    - Yellowing (jaundice indicator): {biomarkers['yellowing']:.4f} (percentage of yellow pixels)
    - Pallor: {biomarkers['pallor']:.4f} (percentage of pale pixels)
    
    Relevant medical information:
    {context}
    
    Based on these biomarkers and medical information, please provide a JSON response with the following structure:
    {{
        "diagnosis": ["Condition 1", "Condition 2", ...],
        "confidence": [85, 60, ...],
        "recommendations": "Recommendations text",
        "disclaimer": "Disclaimer text"
    }}
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a medical AI assistant analyzing facial biomarkers to detect potential health conditions."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        response_text = chat_completion.choices[0].message.content
        return json.loads(response_text)
    except Exception as e:
        st.error(f"Error querying Groq API: {str(e)}")
        return {
            "diagnosis": ["API Error - Using fallback response"],
            "confidence": [0],
            "recommendations": "Please try again or consult a healthcare professional.",
            "disclaimer": "This is a fallback response due to an API error."
        }

# Sidebar setup
st.sidebar.title("Groq API Setup")
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if api_key:
    os.environ["GROQ_API_KEY"] = api_key
    st.sidebar.success("API Key set successfully!")

# Upload image
st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Main app
col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1.image(image, caption="Uploaded Image", use_column_width=True)
    img_array = np.array(image)
    
    if st.sidebar.button("Analyze Image"):
        if not os.getenv("GROQ_API_KEY") and not st.secrets.get("GROQ_API_KEY"):
            st.error("Please enter your Groq API Key in the sidebar before analyzing.")
        else:
            with st.spinner("Extracting facial biomarkers..."):
                biomarkers, face = extract_facial_biomarkers(img_array)
                
                if biomarkers is None:
                    st.error(face)
                else:
                    st.session_state.biomarker_results = biomarkers
                    
                    if face is not None:
                        col1.image(face, caption="Detected Face", use_column_width=True)
                    
                    with st.spinner("Retrieving relevant medical information..."):
                        medical_context = retrieve_medical_info(biomarkers)
                    
                    with st.spinner("Analyzing with Llama-3.3-70b via Groq..."):
                        llm_response = query_llm_with_groq(biomarkers, medical_context)
                        
                        if llm_response:
                            st.session_state.diagnosis = llm_response.get("diagnosis", ["No diagnosis available"])
                            st.session_state.confidence = llm_response.get("confidence", [0])
                            st.session_state.recommendations = llm_response.get("recommendations", "No recommendations available")
                            st.session_state.disclaimer = llm_response.get("disclaimer", "No disclaimer available")

# Display results
if st.session_state.biomarker_results is not None:
    col2.subheader("Extracted Biomarkers")
    biomarker_df = pd.DataFrame({
        'Biomarker': list(st.session_state.biomarker_results.keys()),
        'Value': list(st.session_state.biomarker_results.values())
    })
    col2.dataframe(biomarker_df)
    
    if st.session_state.diagnosis is not None:
        col2.subheader("Potential Conditions")
        for condition, confidence in zip(st.session_state.diagnosis, st.session_state.confidence):
            col2.write(f"{condition} (Confidence: {confidence:.1f}%)")
        
        col2.subheader("Recommendations")
        col2.write(st.session_state.recommendations)
        
        col2.subheader("Disclaimer")
        col2.write(st.session_state.disclaimer)
else:
    col2.write("Upload an image and click 'Analyze Image' to begin.")

# Footer
st.sidebar.markdown("---")
st.sidebar.subheader("About This System")
st.sidebar.info(
    "This application uses computer vision to extract facial biomarkers and "
    "Llama-3.3-70b-versatile via Groq to analyze potential health conditions."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Setup Instructions")
st.sidebar.markdown(
    """
    1. Install required packages:
    ```
    pip install streamlit opencv-python numpy pandas pillow torch facenet-pytorch groq langchain-community sentence-transformers faiss-cpu python-dotenv
    ```
    2. Create a .env file with your Groq API key:
    ```
    GROQ_API_KEY=your_api_key_here
    ```
    3. Get a Groq API key from [Groq Console](https://console.groq.com)
    4. Run the app:
    ```
    streamlit run app.py
    ```
    """
)