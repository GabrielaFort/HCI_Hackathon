import json

def extract_oncotree_names(tissue_name):
    """
    Extracts oncotree names from a file containing multiple JSON objects
    and writes them to an output file, one per line.
    """
    import json

    input_path = f"../data/oncotree_tissues/{tissue_name}.json"
    output_path = f"../data/oncotree_tissues/{tissue_name}_oncotree_names.txt"

    oncotree_names = []

    with open(input_path, "r") as f:
        buffer = ""
        brace_count = 0

        for line in f:
            buffer += line
            brace_count += line.count("{") - line.count("}")

            # When braces balance, we have one complete JSON object
            if brace_count == 0 and buffer.strip():
                obj = json.loads(buffer)
                if "oncotree_name" in obj:
                    oncotree_names.append(obj["oncotree_name"].strip())
                buffer = ""

    # Write output
    with open(output_path, "w") as f:
        for name in oncotree_names:
            f.write(name + "\n")

    print(f"Wrote {len(oncotree_names)} oncotree names to {output_path}")


if __name__ == "__main__":
    tissues = []
    with open("../data/tissue_types.txt", 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                tissues.append(line)
    
    for tissue in tissues:
        extract_oncotree_names(tissue)
