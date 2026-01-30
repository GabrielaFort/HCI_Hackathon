import json
import os
from urllib import response
import ollama


# ---------- File parsing helpers ----------
def parse_lines_file(path):
    # Read a text file where each line is a non-empty string.
    # Return list of strings.
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                lines.append(ln)
    return lines


def parse_oncotree_list(tissue_name, base_path = "../data/oncotree_tissues"):
    """
    Read file ../data/oncotree_tissues/{tissue_name}_oncotree_names.txt
    and return list of oncotree names (one per line).
    """
    tissue_name = tissue_name.strip()
    oncotree_list_path = os.path.join(base_path, f"{tissue_name}_oncotree_names.txt")
    return parse_lines_file(oncotree_list_path)


def parse_tissue_list(tissue_list_path):
    """
    Read a text file where each line is a tissue name.
    """
    return parse_lines_file(tissue_list_path)


def get_tumor_json(tumor_json_path):
    """
    Return raw JSON string (useful to feed to the LLM).
    """
    if not os.path.exists(tumor_json_path):
        raise FileNotFoundError(f"File not found: {tumor_json_path}")
    with open(tumor_json_path, "r", encoding="utf-8") as f:
        return f.read()
    
    
def load_oncotree_name_to_code(tissue_name, data_base_path):
    """
    Loads oncotree_name -> oncotree_code mapping for a tissue.
    """
    import json

    path = f"{data_base_path}/{tissue_name}_oncotree_map.json"

    with open(path, "r") as f:
        return json.load(f)


# ---------- Prompt builders ----------
def create_system_prompt_for_names() -> str:
    return (
        """
        ROLE:
You are an expert pathologist familiar with the OncoTree classification system.

I will give you a list of accepted OncoTree names and a JSON object containing information from a sample pathology report.
Your task is to pick the single OncoTree name from the provided list that best matches the sample.

RULES:
- The list of OncoTree names is delimited by the "$" character.
- Each OncoTree name is exactly the text between two "$" characters.
- You must output the FULL OncoTree name exactly as it appears in the list.
- Do not modify, shorten, or paraphrase the name.
- Only output names that appear in the provided list.
- Output ONLY the selected OncoTree name.
- Do **not** include explanations, reasoning, formatting, or additional text.
- If no appropriate match exists, output: Unknown.

EXAMPLES OF VALID OUTPUT:
Chronic Lymphocytic Leukemia/Small Lymphocytic Lymphoma
High-Grade B-Cell Lymphoma, with MYC and BCL2 and/or BCL6 Rearrangements
        """
    )

def create_user_prompt_for_names(tumor_json, oncotree_names_list):
    return (
        "Tumor Sample JSON:\n"
        f"{tumor_json}\n\n"
        "Oncotree Names:\n"
        f"{'$'.join(oncotree_names_list)}"
    )

def create_system_prompt_for_tissues(tissues):
    return (
        f"""
You are an expert pathologist specializing in tumor classification.

RULES:
- Select the single tissue **from the provided list** that best matches the tumor description.
- If the path report indicates a tumor type or ICD/ICD-O codes indicate a primary tumor tissue, choose that primary tissue. Use sample_site only **only** if no other tissue is indicated.
- Use common synonyms to match tissues when appropriate (e.g., Colon → Bowel).
- Do NOT select tissues that are not in the provided list.
- Respond with ONLY the tissue name exactly as it appears in the provided list.
- No explanations, analysis, quotation marks, or extra text.
- If there is no confident match, respond with: Unknown.

LIST OF TISSUES:
        {tissues}
        """)

# ---------Clean up LLM output ----------
def clean_response(text):
    """
    Return `text` stripped of surrounding whitespace and any single matching
    pair of surrounding quotes:
      - single quote: '...'
      - double quote: "..."
      - backtick: `...`
      - curly quotes: “...” or ‘...’
      - triple quotes: '''...''' or ""...""
    Leaves internal quotes alone. If text is None, returns None.
    """
    if text is None:
        return None

    s = text.strip()

    # Handle triple quotes first ('''text''' or """text""")
    if (len(s) >= 6) and (
        (s.startswith("'''") and s.endswith("'''")) or (s.startswith('"""') and s.endswith('"""'))
    ):
        return s[3:-3].strip()

    # Pairs of opening -> closing quotes to consider
    pairs = {
        "'": "'",
        '"': '"',
        "`": "`",
        "“": "”",
        "‘": "’",
    }

    for open_q, close_q in pairs.items():
        if s.startswith(open_q) and s.endswith(close_q):
            # remove a single surrounding pair and trim again
            return s[len(open_q):-len(close_q)].strip()

    # If nothing matched, just return trimmed text
    return s

# ---------- LLM wrapper ----------
def generate_response(model,temperature,system_prompt,user_prompt):
    """
    Call ollama.chat and return the assistant content string.
    Raises RuntimeError if the response doesn't contain expected structure.
    """
    options = {"temperature": float(temperature)}
    # Ollama client usage assumed available in environment
    response = ollama.chat(
        model=model,
        stream=False,
        options=options,
        think=False,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    # Expected structure: {'message': {'content': '...'}}
    try:
        raw=response['message']['content']
    except (KeyError, TypeError):
        raise RuntimeError(f"Unexpected response structure: {response}")
    
    # Clean raw output
    return clean_response(raw)

# ---------- Convenience / combined flows ----------
def predict_oncotree_name_from_tissue(tissue_name,
                                      tumor_json_path,
                                      model = "granite4:latest",
                                      temperature = 0.0,
                                      data_base_path = "../data/oncotree_tissues"):
    """
    Load oncotree names for a given tissue and tumor json, call LLM, return predicted oncotree name.
    """
    oncotree_names = parse_oncotree_list(tissue_name, base_path=data_base_path)
    tumor_json = get_tumor_json(tumor_json_path)
    sys_prompt = create_system_prompt_for_names()
    user_prompt = create_user_prompt_for_names(tumor_json, oncotree_names)
    return generate_response(model=model, temperature=temperature, system_prompt=sys_prompt, user_prompt=user_prompt)

def predict_tissue_from_list(tissue_list_path,
                             tumor_json_path,
                             model = "granite4:latest",
                             temperature = 0.0):
    """
    Load tissue list and tumor json, call LLM, return predicted tissue.
    """
    tissues = parse_tissue_list(tissue_list_path)
    tumor_json = get_tumor_json(tumor_json_path)
    sys_prompt = create_system_prompt_for_tissues(tissues)
    return generate_response(model=model, temperature=temperature, system_prompt=sys_prompt, user_prompt=tumor_json)



