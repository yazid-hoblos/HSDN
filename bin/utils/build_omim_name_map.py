"""
Build OMIM ID → disease name mapping from a backbone or edge list.

- Extract unique OMIM IDs from an input file (CSV/TSV) with columns Source,Target.
- Resolve names from data/HPO/phenotype.hpoa (columns: database_id, disease_name).
- Save mapping to TSV: data/replication/omim_name_map.tsv

Usage:
  python bin/build_omim_name_map.py \
    --input data/replication/disease_disease_network_backbone.csv \
    --hpo data/HPO/phenotype.hpoa \
    --output data/replication/omim_name_map.tsv
"""

import argparse
from pathlib import Path
import pandas as pd


def extract_omim_ids(input_path: Path, sep: str = ','):
    df = pd.read_csv(input_path, sep=sep)
    ids = set()
    for col in ['Source', 'Target']:
        if col not in df.columns:
            raise ValueError(f"Input missing required column: {col}")
        for val in df[col].astype(str).values:
            val = val.strip()
            if not val:
                continue
            if val.startswith('OMIM:'):
                ids.add(val)
            else:
                # accept bare IDs if present
                try:
                    int_val = int(val)
                    ids.add(f'OMIM:{int_val}')
                except Exception:
                    # ignore non-OMIM entries
                    pass
    return sorted(ids)


def load_hpo_names(hpo_path: Path):
    """Load OMIM→name mapping from phenotype.hpoa."""
    if not hpo_path.exists():
        return {}
    
    # Read TSV, skip comment lines
    df = pd.read_csv(hpo_path, sep='\t', comment='#', dtype=str)
    
    # Columns: database_id, disease_name
    if 'database_id' not in df.columns or 'disease_name' not in df.columns:
        raise ValueError(f"HPO file missing required columns: database_id, disease_name")
    
    # Build mapping: keep first occurrence per OMIM ID
    mapping = {}
    for _, row in df.iterrows():
        db_id = str(row['database_id']).strip()
        if db_id.startswith('OMIM:') and db_id not in mapping:
            mapping[db_id] = str(row['disease_name']).strip()
    
    return mapping


def build_mapping(ids, hpo_names: dict):
    """Build mapping DataFrame from IDs and HPO names."""
    rows = []
    for omim in ids:
        name = hpo_names.get(omim, '')
        rows.append({'omim_id': omim, 'name': name})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True, help='Backbone/edge list with Source,Target columns')
    parser.add_argument('--hpo', type=Path, default=Path('data/HPO/phenotype.hpoa'), help='HPO phenotype.hpoa file')
    parser.add_argument('--output', type=Path, default=Path('data/replication/omim_name_map.tsv'))
    parser.add_argument('--sep', type=str, default=',', help='Input delimiter (default comma)')
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found")
        return

    print("Extracting OMIM IDs...")
    ids = extract_omim_ids(args.input, sep=args.sep)
    print(f"Found {len(ids)} unique OMIM IDs")

    print(f"Loading names from {args.hpo}...")
    hpo_names = load_hpo_names(args.hpo)
    print(f"Loaded {len(hpo_names)} OMIM names from HPO")

    df = build_mapping(ids, hpo_names)
    
    # Count matches
    matched = (df['name'] != '').sum()
    print(f"Matched {matched}/{len(ids)} IDs with names")
    
    # Save TSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, sep='\t', index=False)
    print(f"Saved mapping to {args.output}")
    df.to_csv(args.output, sep='\t', index=False)
    print(f"Saved mapping to {args.output}")

if __name__ == '__main__':
    main()
