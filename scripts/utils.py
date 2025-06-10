import pandas as pd
import re
from pathlib import Path

def parse_hgvs_file(input_csv: Path, output_csv: Path = None, hgvs_column: str = "hgvs_pro",
                    activity_column: str = "score", mapping: dict = None) -> Path:
    if mapping is None:
        mapping = {
            "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
            "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
            "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
            "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
            "Sec": "U"
        }
    df_raw = pd.read_csv(input_csv)
    rows = []
    pattern = re.compile(r"([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")
    for _, row in df_raw.iterrows():
        hgvs_str = row.get(hgvs_column)
        if pd.isna(hgvs_str):
            continue
        hgvs_str = hgvs_str.replace("p.", "")
        m = pattern.match(hgvs_str)
        if not m:
            continue
        wt_3, pos_str, mut_3 = m.groups()
        pos = int(pos_str)
        if wt_3 not in mapping or mut_3 not in mapping:
            continue
        wt_aa = mapping[wt_3]
        mut_aa = mapping[mut_3]
        activity = row.get(activity_column)
        variant_str = f"{wt_aa}{pos}{mut_aa}"
        rows.append({
            "Position": pos,
            "WT_Residue": wt_aa,
            "Mut_Residue": mut_aa,
            "Activity": activity,
            "variant": variant_str
        })
    df_parsed = pd.DataFrame(rows)
    if output_csv is None:
        output_csv = input_csv.parent / f"{input_csv.stem}_parsed.csv"
    df_parsed.to_csv(output_csv, index=False)
    print("Save to", output_csv)
    return output_csv


def read_fasta_as_dict(fasta_file):

    result = {}
    with open(fasta_file, 'r') as f:
        seq_name = None
        seq_str = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):

                if seq_name is not None and seq_str:
                    result[seq_name] = "".join(seq_str)
                seq_name = line[1:]
                seq_str = []
            else:
                seq_str.append(line)
        if seq_name is not None and seq_str:
            result[seq_name] = "".join(seq_str)
    return result

