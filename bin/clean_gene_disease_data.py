"""
Clean and parse gene-disease association data.

This script processes raw ClinVar data which often has multiple diseases
listed together, separated by pipes (|). It splits them into individual
disease-gene associations.
"""

import pandas as pd
from pathlib import Path
import re


def clean_gene_disease_data(input_path, output_path):
    """
    Parse and clean gene-disease association data from ClinVar.
    
    ClinVar often lists multiple diseases for a single gene-variant combination,
    separated by pipes (|). This function splits them into individual associations.
    
    Parameters:
    -----------
    input_path : str or Path
        Input file path
    output_path : str or Path
        Output file path
    """
    print("="*60)
    print("CLEANING GENE-DISEASE DATA")
    print("="*60)
    
    # Load data
    df = pd.read_csv(input_path, sep='\t')
    print(f"\nLoaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Split pipe-separated diseases into individual rows
    cleaned_rows = []
    
    for idx, row in df.iterrows():
        disease_str = str(row['disease'])
        gene = str(row['gene']).strip()
        source = str(row['source']).strip() if 'source' in row else 'ClinVar'
        
        # Split by pipe
        diseases = disease_str.split('|')
        
        for disease in diseases:
            disease = disease.strip()
            
            # Skip empty values and 'not provided'
            if not disease or disease.lower() == 'not provided':
                continue
            
            # Remove common non-disease phrases
            disease = re.sub(r'^\d+$', '', disease).strip()  # Remove pure numbers
            if not disease:
                continue
            
            # Clean up common patterns
            disease = disease.replace('  ', ' ')  # Multiple spaces to single
            disease = disease.strip()
            
            if disease and gene:
                cleaned_rows.append({
                    'disease': disease,
                    'gene': gene,
                    'source': source
                })
        
        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1} rows...")
    
    # Create DataFrame
    cleaned_df = pd.DataFrame(cleaned_rows)
    
    print(f"\nAfter cleaning:")
    print(f"  Total associations: {len(cleaned_df)}")
    print(f"  Unique diseases: {cleaned_df['disease'].nunique()}")
    print(f"  Unique genes: {cleaned_df['gene'].nunique()}")
    
    # Remove duplicates
    cleaned_df = cleaned_df.drop_duplicates()
    
    print(f"\nAfter deduplication:")
    print(f"  Total associations: {len(cleaned_df)}")
    print(f"  Unique diseases: {cleaned_df['disease'].nunique()}")
    print(f"  Unique genes: {cleaned_df['gene'].nunique()}")
    
    # Show sample
    print(f"\nSample of cleaned data:")
    print(cleaned_df.head(10).to_string())
    
    # Save
    cleaned_df.to_csv(output_path, sep='\t', index=False)
    print(f"\nCleaned data saved to: {output_path}")
    
    return cleaned_df


def remove_rare_diseases(input_path, output_path, min_genes=1):
    """
    Optionally filter out very rare diseases (associated with only 1 gene).
    
    Parameters:
    -----------
    input_path : str or Path
        Input file path
    output_path : str or Path
        Output file path
    min_genes : int
        Minimum number of genes a disease must have (default: 1 = no filtering)
    """
    if min_genes <= 1:
        # No filtering needed
        import shutil
        shutil.copy(input_path, output_path)
        return None
    
    df = pd.read_csv(input_path, sep='\t')
    
    # Count genes per disease
    disease_gene_counts = df.groupby('disease')['gene'].nunique()
    
    # Filter
    diseases_to_keep = disease_gene_counts[disease_gene_counts >= min_genes].index
    filtered_df = df[df['disease'].isin(diseases_to_keep)]
    
    print(f"\nFiltering diseases with < {min_genes} genes:")
    print(f"  Before: {len(df)} associations, {df['disease'].nunique()} diseases")
    print(f"  After: {len(filtered_df)} associations, {filtered_df['disease'].nunique()} diseases")
    
    filtered_df.to_csv(output_path, sep='\t', index=False)
    
    return filtered_df


def main():
    """Main execution."""
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'replication'
    
    input_file = data_dir / 'gene_disease_associations.tsv'
    cleaned_file = data_dir / 'gene_disease_associations_cleaned.tsv'
    
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        return
    
    # Clean data
    cleaned_df = clean_gene_disease_data(input_file, cleaned_file)
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print(f"\nYour cleaned data is ready at:")
    print(f"  {cleaned_file}")
    print(f"\nYou can now build the gene-based disease network:")
    print(f"  python bin/gene_disease_network.py")
    print("\nOr update the gene_disease_network.py to use the cleaned file:")
    print(f"  user_file = data_dir / 'gene_disease_associations_cleaned.tsv'")


if __name__ == "__main__":
    main()
