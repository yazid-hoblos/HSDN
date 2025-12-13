"""
Filter significant Symptom–Disease associations using chi-squared (χ²) test.

Inputs:
- data/replication/data3.txt (columns: MeSH Symptom Term, MeSH Disease Term, PubMed occurrence, TFIDF score)

Outputs:
- data/replication/symptom_disease_significant.tsv
- Console summary stats

Usage:
    python bin/filter_symptom_disease_significance.py --n-docs 849103 --p-threshold 0.05

Notes:
- Uses Yates' correction for 2×2 contingency tables.
- Ensures non-negative cells; skips invalid tables.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


def compute_marginals(df):
    # Sum co-occurrences per symptom and per disease
    symptom_totals = df.groupby('MeSH Symptom Term')['PubMed occurrence'].sum().to_dict()
    disease_totals = df.groupby('MeSH Disease Term')['PubMed occurrence'].sum().to_dict()
    return symptom_totals, disease_totals


def contingency(a, sym_total, dis_total, n_docs):
    # a: co-occurrence
    b = sym_total - a
    c = dis_total - a
    d = n_docs - (a + b + c)  # = n_docs - sym_total - dis_total + a
    return a, b, c, d


def filter_significant(df, n_docs, p_threshold=0.05):
    symptom_totals, disease_totals = compute_marginals(df)

    results = []
    skipped = 0

    # Iterate rows; data3.txt can be large, so process in chunks-like loop
    for idx, row in df.iterrows():
        s = row['MeSH Symptom Term']
        d = row['MeSH Disease Term']
        a = int(row['PubMed occurrence'])

        sym_total = symptom_totals.get(s, 0)
        dis_total = disease_totals.get(d, 0)

        a, b, c, d_cell = contingency(a, sym_total, dis_total, n_docs)

        # Validate non-negative and not all zeros
        if min(a, b, c, d_cell) < 0:
            skipped += 1
            continue
        if (a + b + c + d_cell) == 0:
            skipped += 1
            continue

        table = np.array([[a, b], [c, d_cell]])
        # Yates' correction for 2×2
        chi2, p, _, _ = chi2_contingency(table, correction=True)

        if p <= p_threshold:
            results.append({
                'mesh_symptom': s,
                'mesh_disease': row['MeSH Disease Term'],
                'cooccurrence': a,
                'symptom_total': sym_total,
                'disease_total': dis_total,
                'n_docs': n_docs,
                'chi2': chi2,
                'p_value': p,
                'tfidf': row.get('TFIDF score', np.nan)
            })

    return pd.DataFrame(results), skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-docs', type=int, required=False, default=849103,
                        help='Total number of PubMed documents considered (default: 849103).')
    parser.add_argument('--p-threshold', type=float, required=False, default=0.05,
                        help='P-value threshold for significance (default: 0.05).')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    data3_file = base_dir / 'data' / 'replication' / 'data3.txt'
    out_file = base_dir / 'data' / 'replication' / 'symptom_disease_significant.tsv'

    if not data3_file.exists():
        print(f'Error: {data3_file} not found.')
        return

    print('='*60)
    print('FILTERING SIGNIFICANT SYMPTOM–DISEASE ASSOCIATIONS (χ²)')
    print('='*60)
    print(f'N_docs: {args.n_docs:,} | P-threshold: {args.p_threshold}')

    # Read with explicit dtypes to avoid memory bloat
    df = pd.read_csv(data3_file, sep='\t', dtype={
        'MeSH Symptom Term': 'string',
        'MeSH Disease Term': 'string',
        'PubMed occurrence': 'int64',
        'TFIDF score': 'float64'
    })

    # Compute
    significant_df, skipped = filter_significant(df, args.n_docs, args.p_threshold)

    # Save
    significant_df.to_csv(out_file, sep='\t', index=False)

    # Summary
    n_symptoms = significant_df['mesh_symptom'].nunique() if not significant_df.empty else 0
    n_diseases = significant_df['mesh_disease'].nunique() if not significant_df.empty else 0
    avg_diseases_per_symptom = (significant_df.groupby('mesh_symptom')['mesh_disease'].nunique().mean()
                                if not significant_df.empty else 0)

    print('\nSummary:')
    print(f'  Significant edges: {len(significant_df):,}')
    print(f'  Symptoms: {n_symptoms:,} | Diseases: {n_diseases:,}')
    print(f'  Avg diseases per symptom: {avg_diseases_per_symptom:,.2f}')
    print(f'  Skipped invalid tables: {skipped:,}')
    print(f'  Output: {out_file}')


if __name__ == '__main__':
    main()
