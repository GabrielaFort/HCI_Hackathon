import ollama

def parse_tissue_list(tissue_list_path):
    # read csv file
    tissues = []
    with open(tissue_list_path, 'r') as f:
        for line in f:
            tissues.append(line.strip())
    return tissues  

def parse_tumor_json(tumor_json_path):
    with open(tumor_json_path, 'r') as f:
        tumor_json = f.read()
    return tumor_json

def create_system_prompt(tissues):
    prompt = f"""
You are an expert pathologist who specializes in tumor classification. 

I would like you to classify tumors for me using the OncoTree classification system from Memorial Sloan Kettering.

Your job is to find which of the 32 OncoTree tissues does the tumor best match.

I will provide you with a list of possible tissues and a description of the tumor in JSON format.

Respond with only the tissue name that best matches the tumor description.

Repond only with the tissue name, do not include any additional text.

It is important that this is a confident response. If you are not confident in your answer, respond with "Unknown".

Here is the list of possible tissues: {tissues}
"""
    return prompt

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
    return response['message']['content']


if __name__ == "__main__":
    tissue_list_path = "../data/tissue_types.txt"
    tumor_json_path = "../data/0JC3KLGK4Z.json"

    tissues = parse_tissue_list(tissue_list_path)
    tumor_json = parse_tumor_json(tumor_json_path)

    system_prompt = create_system_prompt(tissues)
    print(system_prompt)
    print(tumor_json)

    tissue_type = generate_response(
        model="granite4:latest",
        temperature=0.0,
        system_prompt=system_prompt,
        user_prompt=tumor_json
    )

    print(f"Predicted Tissue Type: {tissue_type}")

