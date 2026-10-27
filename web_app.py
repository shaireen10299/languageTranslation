import streamlit as st
import json
import os
import base64
from model import translate_document

# --- Page Configuration ---
st.set_page_config(
    page_title="BEL Document Translator", 
    page_icon="📄", 
    layout="wide" # Set to 'wide' layout to give enough room for side-by-side view!
)

# --- Sidebar (Branding & Logo) ---
with st.sidebar:
    if os.path.exists("bel_logo.png"):
        st.image("bel_logo.png", use_container_width=True)
    else:
        st.markdown("### [BEL LOGO]")
        
    st.markdown("### Bharat Electronics Limited")
    # st.markdown("**Language Translation System**")
    st.markdown("---")
    st.markdown("Status: **Running Locally** 🟢")

# --- Main Interface ---
st.title("Document Translator")
# st.markdown("Secure, offline document translation")

# 1. Load Languages securely and cache them
@st.cache_data
def load_langs():
    with open("languages.json", "r", encoding="utf-8") as f:
        return json.load(f)

try:
    languages_dict = load_langs()
    language_names = list(languages_dict.keys())
except Exception as e:
    st.error(f"Could not load languages.json: {e}")
    languages_dict = {}
    language_names = []

# 2. File Upload Widget
uploaded_file = st.file_uploader("1. Upload Input English Document (PDF)", type=["pdf"])

# 3. Language Dropdown
target_lang = st.selectbox("2. Select Target Language", ["Select Language..."] + language_names)

# 4. Convert Button & Processing Logic
if st.button("Convert Document", type="primary"):
    if not uploaded_file:
        st.warning("Please upload a PDF file first.")
    elif target_lang == "Select Language...":
        st.warning("Please select a valid target language.")
    else:
        with st.spinner("Translating... Please wait."):
            
            temp_input_path = "temp_input.pdf"
            input_bytes = uploaded_file.getbuffer()
            with open(temp_input_path, "wb") as f:
                f.write(input_bytes)
            
            target_code = languages_dict[target_lang]
            output_filename = f"translated_{target_lang.replace(' ', '_')}.pdf"
            output_filepath = os.path.join("Output", output_filename)
            
            try:
                # Trigger the translation pipeline
                translate_document(temp_input_path, output_filepath, target_code, target_lang)
                
                st.success("🎉 Document translated successfully!")
                
                # --- HIDDEN ACCURACY REPORT ACCORDION ---
                eval_file_path = f"./Evaluation/evaluation_{target_lang.replace(' ', '_')}.txt"
                if os.path.exists(eval_file_path):
                    with open(eval_file_path, "r", encoding="utf-8") as eval_f:
                        eval_report_text = eval_f.read()
                    
                    with st.expander("📊 View Mathematical & Semantic Accuracy Report"):
                        st.text(eval_report_text)
                
                # Read the translated PDF data
                with open(output_filepath, "rb") as pdf_file:
                    output_pdf_data = pdf_file.read()
                
                # Download Button for convenience
                st.download_button(
                    label="⬇️ Download Translated PDF",
                    data=output_pdf_data,
                    file_name=output_filename,
                    mime="application/pdf",
                    type="primary"
                )
                
                st.divider()
                st.markdown("### 🔍 Document Comparison Preview")
                
                # --- SIDE-BY-SIDE LAYOUT USING COLUMNS ---
                col1, col2 = st.columns(2)
                
                # Left Column: Original Uploaded PDF Preview
                with col1:
                    st.markdown("#### 📄 Original Document (English)")
                    base64_input = base64.b64encode(input_bytes).decode('utf-8')
                    input_display = f'<iframe src="data:application/pdf;base64,{base64_input}" width="100%" height="700" type="application/pdf"></iframe>'
                    st.markdown(input_display, unsafe_allow_html=True)
                
                # Right Column: Translated PDF Preview
                with col2:
                    st.markdown(f"#### 🌐 Translated Document ({target_lang})")
                    base64_output = base64.b64encode(output_pdf_data).decode('utf-8')
                    output_display = f'<iframe src="data:application/pdf;base64,{base64_output}" width="100%" height="700" type="application/pdf"></iframe>'
                    st.markdown(output_display, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"An error occurred during translation: {e}")
            finally:
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)