import streamlit as st
import ollama
import json

# ---------- core logic (same as your script) ----------
def parse_tissue_list(tissue_list_path):
    tissues = []
    with open(tissue_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                tissues.append(line)
    return tissues

def create_system_prompt(tissues):
    return f"""
You are an expert pathologist who specializes in tumor classification.

I would like you to classify tumors for me using the OncoTree classification system from Memorial Sloan Kettering.

Your job is to find which of the 32 OncoTree tissues the tumor best matches.

I will provide you with a list of possible tissues and a description of the tumor in JSON format.

Respond with only the tissue name that best matches the tumor description.

Respond only with the tissue name; do not include any additional text.

If you are not confident, respond with "Unknown". 

Here is the list of possible tissues: {tissues}
"""

def generate_response(model, temperature, system_prompt, user_prompt):
    response = ollama.chat(
        model=model,
        stream=False,
        options={"temperature": temperature},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response["message"]["content"].strip()

# ---------- streamlit UI ----------

st.set_page_config(page_title="Tumor → Tissue Classifier", layout="centered")

st.title("🧬 Tumor Tissue Classifier")
st.caption("Upload a tumor JSON and get the predicted OncoTree tissue")

# load tissues once
@st.cache_data
def load_tissues():
    return parse_tissue_list("../data/tissue_types.txt")

tissues = load_tissues()
system_prompt = create_system_prompt(tissues)

model = st.text_input("Model", value="granite4:latest")

uploaded_file = st.file_uploader(
    "Upload tumor JSON",
    type=["json"]
)

if uploaded_file is not None:
    try:
        # Read and pretty-print JSON for display + prompt
        tumor_obj = json.load(uploaded_file)
        tumor_json_str = json.dumps(tumor_obj, indent=2)

        st.subheader("Tumor JSON (preview)")
        st.code(tumor_json_str, language="json")

        if st.button("Classify Tumor"):
            with st.spinner("Classifying..."):
                tissue_type = generate_response(
                    model=model,
                    temperature=0.0,
                    system_prompt=system_prompt,
                    user_prompt=tumor_json_str
                )

            st.success("Prediction complete")
            st.markdown(f"### 🧪 Predicted Tissue Type\n**{tissue_type}**")

    except json.JSONDecodeError:
        st.error("Invalid JSON file. Please upload a valid tumor JSON.")