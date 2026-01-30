#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import oncotree_utils as tools

def get_test_order_id(parsed, filename):
    return parsed.get("test_order_id") 

def process_file(path, tissue_list, oncotree_base, model, temperature, mode="local"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            parsed = json.load(f)
    except Exception:
        parsed = {}

    test_order_id = get_test_order_id(parsed, path)

    # predict tissue
    try:
        tissue = tools.predict_tissue_from_list(
            tissue_list_path=tissue_list,
            tumor_json_path=path,
            model=model,
            temperature=temperature,
            mode=mode,
        ).strip()
    except Exception:
        tissue = "none"
    if not tissue or tissue.lower() == "unknown":
        tissue = "none"

    # predict oncotree name
    try:
        onco_name = tools.predict_oncotree_name_from_tissue(
            tissue_name=tissue,
            tumor_json_path=path,
            model=model,
            temperature=temperature,
            data_base_path=oncotree_base,
            mode=mode
        ).strip()
    except Exception:
        onco_name = "none"
    if not onco_name:
        onco_name = "none"

    # map name -> code
    onco_code = "none"
    if tissue and onco_name:
        try:
            mapping = tools.load_oncotree_name_to_code(tissue_name=tissue, data_base_path=oncotree_base)
            onco_code = mapping.get(onco_name, "none") or "none"
        except Exception:
            onco_code = "none"

    return {
        "oncotree_tissue": tissue,
        "oncotree_code": onco_code,
        "oncotree_name": onco_name,
        "test_order_id": test_order_id,
        "confidence": 5, # placeholder confidence
        "rationale": "" # placeholder rationale

    }

def main():
    p = argparse.ArgumentParser(description="Batch run OncoTree predictions and write JSONL")
    p.add_argument("--input-dir", required=True, help="Directory with .json tumor files")
    p.add_argument("--output", required=True, help="Output JSONL file")
    p.add_argument("--tissue-list", default="../data/tissue_types.txt", help="Path to tissue_types.txt")
    p.add_argument("--oncotree-base", default="../data/oncotree_tissues", help="Base dir for oncotree mappings")
    p.add_argument("--model", default="granite4:latest", help="Model name")
    p.add_argument("--temperature", type=float, default=0.0, help="Model temperature")
    p.add_argument("--mode", choices=["local", "cloud"], default="local", help="Whether to run cloud or local ollama model")
    p.add_argument("--ext", default=".json", help="File extension to look for")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print("Input directory missing:", input_dir)
        return

    files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == args.ext.lower()])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a", encoding="utf-8") as out:
        for f in files:
            try:
                res = process_file(str(f), args.tissue_list, args.oncotree_base, args.model, args.temperature, mode=args.mode)
            except Exception:
                res = {
                    "oncotree_tissue": "",
                    "oncotree_code": "",
                    "oncotree_name": "",
                    "test_order_id": f.stem,
                }
            out.write(json.dumps(res) + "\n")
            out.flush()
            print("Wrote:", f.name)

if __name__ == "__main__":
    main()
