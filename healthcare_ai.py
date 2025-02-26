import streamlit as st
from PyPDF2 import PdfReader
import fitz  # PyMuPDF for better image extraction
import google.generativeai as genai
import io
import base64
import json

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyAYfcTAFba5mn5LXw4UNNfnBvQEgmNbAos"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')  # Using Gemini 1.5 Flash

class HealthcareAgent:
    def analyze_with_gemini(self, image_bytes, prompt):
        """Sends image to Gemini API for medical analysis"""
        try:
            # Convert image to Base64
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")

            response = model.generate_content([
                {
                    "role": "user",
                    "parts": [
                        {"inline_data": {"mime_type": "image/png", "data": encoded_image}},  # Correct format for image
                        {"text": prompt}  # Correct format for text
                    ]
                }
            ])
            return response.text
        except Exception as e:
            return f"API Error: {str(e)}"

    def extract_images_from_pdf(self, uploaded_pdf):
        """Extracts images from a PDF (Medical Report)"""
        try:
            images = []
            pdf_document = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")

            for page in pdf_document:
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    images.append(image_bytes)

            return images  # Return images list (empty if no images found)
        except Exception as e:
            return []

    def extract_text_from_pdf(self, uploaded_pdf):
        """Extracts text from a PDF for analysis if no images are found"""
        try:
            reader = PdfReader(uploaded_pdf)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip() if text.strip() else None
        except Exception as e:
            return None
    def symptom_checker(self, symptoms):
        """Checks symptoms and suggests conditions"""
        prompt = f"""Analyze these symptoms: {symptoms}
        Provide response in STRICT JSON format:
        {{
            "possible_conditions": ["Example: Flu"],
            "recommended_actions": ["Example: Rest and drink fluids"],
            "emergency_signs": ["Example: Seek medical attention immediately"]
        }}
        Ensure output is pure JSON (no markdown or extra text)."""

        try:
            response = model.generate_content([
                {"role": "user", "parts": [{"text": prompt}]}  # Correct format
            ])
            response_text = response.text.replace("```json", "").replace("```", "").strip()  # Clean response
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {"error": "Failed to parse medical analysis"}
        except Exception as e:
            return {"error": str(e)}

    def medication_analyzer(self, medications):
        """Analyzes medications for interactions, side effects, and guidelines"""
        if not medications.strip():
            return {"error": "No medication input provided"}

        prompt = f"""
        Analyze these medications: {medications}
        Provide response in STRICTLY VALID JSON format:
        {{
            "interactions": ["Example: Drug A may interact with Drug B"],
            "side_effects": ["Example: May cause drowsiness"],
            "guidelines": ["Example: Take after food"]
        }}
        Ensure output is pure JSON (no markdown or extra text)."""

        try:
            response = model.generate_content([
                {"role": "user", "parts": [{"text": prompt}]}  # Correct format
            ])
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            return {"error": "Failed to parse medication analysis"}
        except Exception as e:
            return {"error": str(e)}

# Streamlit UI Configuration
st.set_page_config(page_title="Healthcare AI Agent", layout="wide")
st.title("🏥 MediAssist - AI Healthcare Assistant")

tab1, tab2, tab3 = st.tabs(["Symptom Checker", "Report Analysis", "Medication Manager"])

# SYMPTOM CHECKER
with tab1:
    st.subheader("Symptom Analysis")
    symptoms = st.text_area("Describe your symptoms (e.g., fever, headache):")

    if st.button("Analyze Symptoms"):
        agent = HealthcareAgent()
        with st.spinner('Analyzing symptoms...'):
            result = agent.symptom_checker(symptoms)

            if "error" in result:
                st.error(f"Analysis failed: {result['error']}")
            else:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("Possible Conditions")
                    for condition in result.get("possible_conditions", [])[:3]:
                        st.markdown(f"- {condition}")

                with col2:
                    st.subheader("Recommended Actions")
                    for action in result.get("recommended_actions", []):
                        st.markdown(f"- {action}")

                with col3:
                    st.subheader("Emergency Signs")
                    for sign in result.get("emergency_signs", []):
                        st.markdown(f"⚠ {sign}")

# MEDICAL REPORT & IMAGE ANALYSIS
with tab2:
    st.subheader("Medical Report & Scan Analysis")

    # PDF Uploader for extracting images
    uploaded_pdf = st.file_uploader("Upload Medical Report (PDF with X-ray/MRI/CT scan images)", type=['pdf'], key="report")

    if uploaded_pdf:
        agent = HealthcareAgent()
        with st.spinner('Processing medical report...'):
            image_list = agent.extract_images_from_pdf(uploaded_pdf)
            extracted_text = agent.extract_text_from_pdf(uploaded_pdf)

        # Display and analyze extracted images
        if image_list:
            for i, img_bytes in enumerate(image_list):
                st.image(img_bytes, caption=f"Extracted Image {i+1}", use_container_width=True)

                with st.spinner('Analyzing medical image...'):
                    prompt = "Analyze this medical scan and detect any abnormalities. Provide possible conditions and medication recommendations."
                    result = agent.analyze_with_gemini(img_bytes, prompt)
                    st.subheader(f"Analysis for Image {i+1}")
                    st.markdown(result)

        # Analyze extracted text only if no images are found
        if extracted_text:
            with st.spinner("Analyzing medical report text..."):
                prompt = f"Analyze the following medical report text:\n\n{extracted_text}\n\nProvide possible conditions and recommendations."
                response = model.generate_content([
                    {"role": "user", "parts": [{"text": prompt}]}
                ])
                st.subheader("Medical Report Analysis")
                st.markdown(response.text)

    # Direct image upload (X-ray, MRI, CT scans) - THIS PART IS UNCHANGED
    uploaded_image = st.file_uploader("Or Upload a Medical Scan Image (X-ray, MRI, CT, etc.)", type=['png', 'jpg', 'jpeg'])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Medical Scan", use_container_width=True)

        agent = HealthcareAgent()
        
        with st.spinner('Analyzing uploaded scan...'):
            image_bytes = uploaded_image.read()
            prompt = "Analyze this medical scan and detect any abnormalities. Provide possible conditions and medication recommendations."
            result = agent.analyze_with_gemini(image_bytes, prompt)
            
            st.subheader("Analysis Result")
            st.markdown(result)

# MEDICATION ANALYSIS
with tab3:
    st.subheader("Medication Analysis")
    meds = st.text_input("Enter medications (comma-separated):")

    if st.button("Analyze Medications"):
        agent = HealthcareAgent()
        with st.spinner('Checking interactions...'):
            result = agent.medication_analyzer(meds)

            if "error" in result:
                st.error(f"Analysis failed: {result['error']}")
            else:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("Interactions")
                    for interaction in result.get("interactions", []):
                        st.markdown(f"- {interaction}")

                with col2:
                    st.subheader("Side Effects")
                    for effect in result.get("side_effects", []):
                        st.markdown(f"- {effect}")

                with col3:
                    st.subheader("Guidelines")
                    for guideline in result.get("guidelines", []):
                        st.markdown(f"- {guideline}")

st.divider()
st.caption("Note: This AI assistant provides informational support only and does not replace professional medical advice.")
