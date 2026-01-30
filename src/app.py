import streamlit as st
import oncotree_utils as tools
import tempfile
import json

st.set_page_config(page_title="AI Oncotree Coder Assistant", layout="centered")
st.title("AI Oncotree Classification Assistant")
st.text("Predict oncotree tissue, name, and code from tumor JSON using locally-run LLMs")

# initiate session state variables (keep defaults)
if "override_confirmed" not in st.session_state:
    st.session_state["override_confirmed"] = False
if "override_confirmed_name" not in st.session_state:
    st.session_state["override_confirmed_name"] = False
if "last_tumor_path" not in st.session_state:
    st.session_state["last_tumor_path"] = None

# LLM settings (kept minimal)
available_models = tools.discover_local_ollama_models()
if not available_models:
    st.sidebar.error(
        "No local Ollama models discovered. Make sure Ollama is installed and running, "
        "and that the Python `ollama` package can reach it."
    )

model = st.sidebar.selectbox("Model", options=available_models or ["(no models)"], index=0)
temperature = st.sidebar.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

def _write_tmp(uploaded, suffix):
    if uploaded is None:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getvalue())
    tmp.close()
    return tmp.name

# Tumor JSON (upload)
uploaded_tumor = st.file_uploader("Upload tumor JSON", type=["json"])

tumor_json_path = None
if uploaded_tumor is not None:
    tumor_json_path = _write_tmp(uploaded_tumor, ".json")

# Reset override confirmations only when a new file is uploaded (so we don't wipe user clicks on every rerun)
if tumor_json_path != st.session_state.get("last_tumor_path"):
    st.session_state["override_confirmed"] = False
    st.session_state["override_confirmed_name"] = False
    st.session_state["last_tumor_path"] = tumor_json_path

if not tumor_json_path:
    st.info("Provide tumor JSON (upload) to proceed.")
    st.stop()

# ------------------ Preview (collapsible) ------------------
try:
    raw = uploaded_tumor.getvalue().decode("utf-8")
except Exception:
    raw = uploaded_tumor.getvalue()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")

try:
    parsed = json.loads(raw)
    with st.expander("Preview tumor JSON (expand to view)", expanded=False):
        st.json(parsed)
        if isinstance(parsed, dict):
            keys = list(parsed.keys())
            st.caption(f"Top-level keys: {', '.join(keys[:20])}{'...' if len(keys)>20 else ''}")
        elif isinstance(parsed, list):
            st.caption(f"JSON is an array of length {len(parsed)}. Showing first element below:")
            st.json(parsed[0] if parsed else {})
except json.JSONDecodeError:
    with st.expander("Preview tumor JSON (raw / failed to parse)", expanded=False):
        st.code(raw[:5000], language="json")
# ------------------------------------------------------------

# Step 1: tissue list -> predict tissue
st.header("Step 1 — Predict tissue")
tissue_list_path = "../data/tissue_types.txt"

with st.spinner("Predicting oncotree tissue..."):
    predicted_tissue = tools.predict_tissue_from_list(
        tissue_list_path=tissue_list_path,
        tumor_json_path=tumor_json_path,
        model=model,
        temperature=temperature
    )

st.subheader("Predicted tissue")
st.code(predicted_tissue or "(empty)")

tissues = tools.parse_tissue_list(tissue_list_path)
# safe default if empty
if not tissues:
    tissues = ["(no tissues found)"]

chosen_tissue = st.selectbox(
    "*Optional: override tissue for step 2*",
    options=tissues,
    index=tissues.index(predicted_tissue) if predicted_tissue in tissues else 0
)

# --- small helper callbacks for buttons (use on_click to set state) ---
def _confirm_override():
    st.session_state["override_confirmed"] = True

def _confirm_override_name():
    st.session_state["override_confirmed_name"] = True

# --- Validation / stopping logic ---
pred_lower = (predicted_tissue or "").strip().lower()
is_blank = (predicted_tissue == "")

# If prediction is explicit "Unknown" -> require explicit override confirmation
if pred_lower == "unknown":
    st.warning(
        "Model could not identify the tissue type — please flag this sample for manual review. "
        "If you want to continue, choose a tissue from the dropdown and click the button below to confirm override."
    )

    # use button with a unique key and on_click callback to set session state
    st.button("Override and continue with selected tissue", key="override_tissue_btn", on_click=_confirm_override)

    if not st.session_state["override_confirmed"]:
        st.info("Waiting for manual override confirmation to continue.")
        st.stop()

# If prediction is blank or prediction not in canonical list -> require explicit override confirmation
if is_blank or (predicted_tissue and predicted_tissue not in tissues):
    st.warning("Model did not return an accepted tissue. Select a tissue from the list and click the button below to confirm override if you want to proceed.")
    st.button("Override and continue with selected tissue", key="override_tissue_btn_2", on_click=_confirm_override)

    if not st.session_state["override_confirmed"]:
        st.info("Waiting for manual override confirmation to continue.")
        st.stop()

st.success(f"Using tissue: {chosen_tissue}")

# Step 2: predict oncotree name using chosen tissue
st.header("Step 2 — Predict oncotree name")
run = st.button("Run Step 2")

if run:
    with st.spinner("Predicting OncoTree name..."):
        onco_pred = tools.predict_oncotree_name_from_tissue(
            tissue_name=chosen_tissue,
            tumor_json_path=tumor_json_path,
            model=model,
            temperature=temperature,
            data_base_path="../data/oncotree_tissues"
        ).strip()
    st.subheader("OncoTree name")
    st.code(onco_pred or "(empty)")

    oncotree_map = tools.load_oncotree_name_to_code(
        tissue_name=chosen_tissue,
        data_base_path="../data/oncotree_tissues"
    )

    canonical_names = list(oncotree_map.keys()) or ["(no canonical names)"]

    onco_lower = (onco_pred or "").strip().lower()
    is_onco_blank = (onco_pred == "")

    if onco_lower == "unknown" or is_onco_blank or (onco_pred and onco_pred not in canonical_names):
        st.warning(
            "Model did not return a valid OncoTree name for this tissue. "
            "Please select a name from the canonical list below and click the button to confirm override."
        )

        chosen_onco_name = st.selectbox(
            "*Select OncoTree name to use for mapping*",
            options=canonical_names,
            index=0
        )

        # name override button with its own key and callback
        st.button("Override and continue with selected OncoTree name", key="override_onco_name_btn", on_click=_confirm_override_name)

        if not st.session_state["override_confirmed_name"]:
            st.info("Waiting for manual override confirmation to continue.")
            st.stop()

        final_onco_name = chosen_onco_name
    else:
        final_onco_name = onco_pred

    # Step 3: map oncotree name to code
    st.header("Step 3 — Map OncoTree name to code")
    oncotree_code = oncotree_map.get(final_onco_name)

    if oncotree_code:
        st.subheader("OncoTree code")
        st.code(oncotree_code)
    else:
        st.warning("OncoTree name not found in lookup table for this tissue.")
