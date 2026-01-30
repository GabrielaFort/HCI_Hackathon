def extract_oncotree_name_to_code(tissue_name):
    """
    Extracts oncotree_name -> oncotree_code mapping
    from a file containing multiple JSON objects.
    """
    import json

    input_path = f"../data/oncotree_tissues/{tissue_name}.json"
    output_path = f"../data/oncotree_tissues/{tissue_name}_oncotree_map.json"

    oncotree_map = {}

    with open(input_path, "r") as f:
        buffer = ""
        brace_count = 0

        for line in f:
            buffer += line
            brace_count += line.count("{") - line.count("}")

            # When braces balance, we have one complete JSON object
            if brace_count == 0 and buffer.strip():
                obj = json.loads(buffer)

                if "oncotree_name" in obj and "oncotree_code" in obj:
                    name = obj["oncotree_name"].strip()
                    code = obj["oncotree_code"].strip()
                    oncotree_map[name] = code

                buffer = ""

    # Write output dictionary to file
    with open(output_path, "w") as f:
        json.dump(oncotree_map, f, indent=2)

    print(f"Wrote {len(oncotree_map)} entries to {output_path}")

    return oncotree_map

if __name__ == "__main__":
    tissues = []
    with open("../data/tissue_types.txt", 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                tissues.append(line)
    
    for tissue in tissues:
        extract_oncotree_name_to_code(tissue)
        