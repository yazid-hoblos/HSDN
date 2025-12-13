"""
Download and Prepare Gene-Disease Association Data

This script helps download and prepare gene-disease associations from:
1. OMIM (Online Mendelian Inheritance in Man)
2. PharmGKB (Pharmacogenomics Knowledge Base)
3. GAD (Genetic Association Database)

Note: Some sources require registration and API keys.
"""

import pandas as pd
import requests
import json
import re
from pathlib import Path
from io import StringIO
import time


def download_disgenet(output_path, email=None):
    """
    Download gene-disease associations from DisGeNET (free alternative to GAD).
    
    DisGeNET is a comprehensive platform integrating gene-disease associations
    from various sources including OMIM, PharmGKB, and many others.
    
    Parameters:
    -----------
    output_path : Path
        Where to save the downloaded data
    email : str, optional
        Your email for the NCBI API (recommended but not required)
    """
    print("\n" + "="*60)
    print("DOWNLOADING FROM DISGENET")
    print("="*60)
    
    # DisGeNET provides curated gene-disease associations
    # Free download available from: https://www.disgenet.org/downloads
    
    print("\nDisGeNET provides free gene-disease association data.")
    print("Visit: https://www.disgenet.org/downloads")
    print("\nSteps:")
    print("1. Go to https://www.disgenet.org/downloads")
    print("2. Download 'curated_gene_disease_associations.tsv.gz'")
    print("3. Save and extract to:")
    print(f"   {output_path}")
    print("\nAlternatively, for programmatic access:")
    print("1. Register at https://www.disgenet.org/signup/")
    print("2. Get API key")
    print("3. Use their REST API")
    
    return None


def parse_omim_morbidmap(filepath, output_path):
    """
    Parse OMIM morbidmap file.
    
    Format: Phenotype | Gene Symbols | MIM Number | Cyto Location
    Example: Alzheimer disease, late-onset | APOE | 104300 | 19q13.2
    
    Parameters:
    -----------
    filepath : Path
        Path to downloaded morbidmap.txt
    output_path : Path
        Where to save parsed data
    """
    print("\n" + "="*60)
    print("PARSING OMIM MORBIDMAP")
    print("="*60)
    
    if not filepath.exists():
        print(f"File not found: {filepath}")
        print("\nTo download OMIM data:")
        print("1. Register at https://www.omim.org/")
        print("2. Request API access at https://www.omim.org/api")
        print("3. Download morbidmap.txt")
        return None
    
    associations = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('|')
            if len(parts) < 3:
                continue
            
            phenotype = parts[0].strip()
            genes = parts[1].strip()
            
            # Skip if no gene
            if not genes:
                continue
            
            # Clean phenotype name (remove MIM numbers and uncertainty markers)
            phenotype = re.sub(r'\s*\(\d+\)', '', phenotype)
            phenotype = re.sub(r'[{}?\[\]]', '', phenotype)
            phenotype = phenotype.strip()
            
            # Split multiple genes if present
            gene_list = re.split(r'[,;]', genes)
            
            for gene in gene_list:
                gene = gene.strip()
                if gene:
                    associations.append({
                        'disease': phenotype,
                        'gene': gene,
                        'source': 'OMIM'
                    })
    
    df = pd.DataFrame(associations)
    print(f"Parsed {len(df)} associations from OMIM")
    print(f"Unique diseases: {df['disease'].nunique()}")
    print(f"Unique genes: {df['gene'].nunique()}")
    
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved to: {output_path}")
    
    return df


def download_pharmgkb_api(api_key=None, output_path=None):
    """
    Download PharmGKB data via API.
    
    Note: PharmGKB requires registration for API access.
    
    Parameters:
    -----------
    api_key : str, optional
        PharmGKB API key
    output_path : Path
        Where to save the data
    """
    print("\n" + "="*60)
    print("PHARMGKB DATA")
    print("="*60)
    
    print("\nPharmGKB data download:")
    print("1. Register at https://www.pharmgkb.org/")
    print("2. Go to https://www.pharmgkb.org/downloads")
    print("3. Download 'relationships.zip'")
    print("4. Extract and look for files:")
    print("   - genes.tsv")
    print("   - diseases.tsv")
    print("   - relationships.tsv (contains gene-disease links)")
    
    return None


def parse_pharmgkb_relationships(filepath, output_path):
    """
    Parse PharmGKB relationships file.
    
    Parameters:
    -----------
    filepath : Path
        Path to PharmGKB relationships.tsv
    output_path : Path
        Where to save parsed data
    """
    print("\n" + "="*60)
    print("PARSING PHARMGKB RELATIONSHIPS")
    print("="*60)
    
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return None
    
    df = pd.read_csv(filepath, sep='\t')
    print(f"Loaded {len(df)} relationships")
    print(f"Columns: {list(df.columns)}")
    
    # Filter for gene-disease relationships
    # PharmGKB format varies, adapt based on actual file
    gene_disease = df[df['Entity1_type'] == 'Gene']
    gene_disease = gene_disease[gene_disease['Entity2_type'] == 'Disease']
    
    associations = []
    for _, row in gene_disease.iterrows():
        associations.append({
            'disease': row['Entity2_name'],
            'gene': row['Entity1_name'],
            'source': 'PharmGKB'
        })
    
    result = pd.DataFrame(associations)
    print(f"Parsed {len(result)} gene-disease associations")
    
    result.to_csv(output_path, sep='\t', index=False)
    print(f"Saved to: {output_path}")
    
    return result


def download_with_entrez(email, output_path):
    """
    Download gene-disease associations using NCBI Entrez E-utilities.
    This uses the free NCBI Gene database.
    
    Parameters:
    -----------
    email : str
        Your email (required by NCBI)
    output_path : Path
        Where to save the data
    """
    print("\n" + "="*60)
    print("DOWNLOADING FROM NCBI GENE")
    print("="*60)
    
    print("\nThis method uses NCBI's free databases.")
    print("It may take a while...")
    
    # Example: Download disease-associated genes from ClinVar
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    # Search for gene-disease relationships in ClinVar
    print("\nQuerying ClinVar for gene-disease associations...")
    
    # This is a simplified example - full implementation would require
    # multiple API calls and parsing
    print("\nFor a complete ClinVar download:")
    print("1. Go to: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/")
    print("2. Download: variant_summary.txt.gz")
    print("3. Extract and parse the file")
    
    return None


def download_clinvar_variant_summary(output_path):
    """
    Download ClinVar variant summary which contains gene-disease information.
    
    Parameters:
    -----------
    output_path : Path
        Where to save the file
    """
    print("\n" + "="*60)
    print("DOWNLOADING CLINVAR DATA")
    print("="*60)
    
    url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
    
    print(f"\nDownloading from: {url}")
    print("This file is large (~500MB compressed). Please wait...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save compressed file
        compressed_path = output_path.parent / "variant_summary.txt.gz"
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(compressed_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='')
        
        print(f"\n\nDownloaded to: {compressed_path}")
        print("Please extract the file using gunzip or similar tool")
        
        return compressed_path
        
    except Exception as e:
        print(f"Error downloading: {e}")
        print("\nManual download:")
        print(f"1. Visit: {url}")
        print("2. Download and extract")
        print(f"3. Save to: {output_path}")
        return None


def parse_clinvar_variant_summary(filepath, output_path):
    """
    Parse ClinVar variant_summary.txt to extract gene-disease associations.
    
    Parameters:
    -----------
    filepath : Path
        Path to variant_summary.txt
    output_path : Path
        Where to save parsed gene-disease associations
    """
    print("\n" + "="*60)
    print("PARSING CLINVAR VARIANT SUMMARY")
    print("="*60)
    
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return None
    
    print("Reading ClinVar data (this may take a few minutes)...")
    
    # Read in chunks due to large file size
    associations = set()  # Use set to avoid duplicates
    
    chunk_size = 100000
    for i, chunk in enumerate(pd.read_csv(filepath, sep='\t', chunksize=chunk_size, 
                                         low_memory=False, encoding='utf-8', 
                                         on_bad_lines='skip')):
        
        # Filter for pathogenic variants with gene and phenotype info
        filtered = chunk[
            (chunk['GeneSymbol'].notna()) & 
            (chunk['PhenotypeList'].notna()) &
            (chunk['ClinicalSignificance'].str.contains('Pathogenic', na=False))
        ]
        
        for _, row in filtered.iterrows():
            gene = str(row['GeneSymbol']).strip()
            phenotypes = str(row['PhenotypeList']).split(';')
            
            for phenotype in phenotypes:
                phenotype = phenotype.strip()
                if phenotype and gene:
                    associations.add((phenotype, gene))
        
        if (i + 1) % 10 == 0:
            print(f"Processed {(i + 1) * chunk_size:,} rows... Found {len(associations):,} associations")
    
    # Convert to DataFrame
    df = pd.DataFrame(list(associations), columns=['disease', 'gene'])
    df['source'] = 'ClinVar'
    
    print(f"\nTotal associations found: {len(df)}")
    print(f"Unique diseases: {df['disease'].nunique()}")
    print(f"Unique genes: {df['gene'].nunique()}")
    
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved to: {output_path}")
    
    return df


def merge_gene_disease_sources(source_files, output_path):
    """
    Merge gene-disease associations from multiple sources.
    
    Parameters:
    -----------
    source_files : list of Path
        List of parsed source files
    output_path : Path
        Where to save merged data
    """
    print("\n" + "="*60)
    print("MERGING DATA SOURCES")
    print("="*60)
    
    all_data = []
    
    for filepath in source_files:
        if filepath.exists():
            df = pd.read_csv(filepath, sep='\t')
            all_data.append(df)
            print(f"Loaded {len(df)} associations from {filepath.name}")
    
    if not all_data:
        print("No data files found to merge")
        return None
    
    # Combine all sources
    merged = pd.concat(all_data, ignore_index=True)
    
    print(f"\nBefore deduplication: {len(merged)} associations")
    
    # Remove duplicates
    merged = merged.drop_duplicates(subset=['disease', 'gene'])
    
    print(f"After deduplication: {len(merged)} associations")
    print(f"Unique diseases: {merged['disease'].nunique()}")
    print(f"Unique genes: {merged['gene'].nunique()}")
    
    # Save merged data
    merged.to_csv(output_path, sep='\t', index=False)
    print(f"\nMerged data saved to: {output_path}")
    
    # Statistics by source
    print("\nAssociations by source:")
    for source in merged['source'].unique():
        count = len(merged[merged['source'] == source])
        print(f"  {source}: {count}")
    
    return merged


def main():
    """Main execution function."""
    
    print("="*60)
    print("GENE-DISEASE DATA DOWNLOAD AND PREPARATION")
    print("="*60)
    
    # Setup directories
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'replication'
    raw_dir = data_dir / 'raw_gene_disease'
    raw_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nData will be saved to: {raw_dir}")
    
    # Method selection
    print("\n" + "="*60)
    print("DATA SOURCE OPTIONS")
    print("="*60)
    print("\n1. ClinVar (NCBI) - FREE, no registration")
    print("   - Comprehensive variant-disease-gene associations")
    print("   - Large file (~500MB)")
    print("   - Automated download available")
    print("\n2. OMIM - Requires registration")
    print("   - Gold standard for Mendelian diseases")
    print("   - Manual download required")
    print("\n3. PharmGKB - Requires registration")
    print("   - Pharmacogenomics focus")
    print("   - Manual download required")
    print("\n4. DisGeNET - FREE")
    print("   - Integrates multiple sources")
    print("   - Manual download required")
    
    choice = input("\nSelect option (1-4) or 'all' for instructions for all: ").strip()
    
    if choice == '1':
        # Download and parse ClinVar
        print("\nDownloading ClinVar data...")
        clinvar_raw = raw_dir / "variant_summary.txt"
        clinvar_parsed = raw_dir / "clinvar_gene_disease.tsv"
        
        # Check if already downloaded
        if not clinvar_raw.exists():
            download_clinvar_variant_summary(clinvar_raw)
            print("\nAfter extracting the .gz file, run this script again to parse it.")
        else:
            parse_clinvar_variant_summary(clinvar_raw, clinvar_parsed)
            
            # Copy to final location
            final_output = data_dir / 'gene_disease_associations.tsv'
            if clinvar_parsed.exists():
                import shutil
                shutil.copy(clinvar_parsed, final_output)
                print(f"\nFinal data ready at: {final_output}")
    
    elif choice == '2':
        # OMIM instructions
        omim_raw = raw_dir / "morbidmap.txt"
        omim_parsed = raw_dir / "omim_gene_disease.tsv"
        
        if omim_raw.exists():
            parse_omim_morbidmap(omim_raw, omim_parsed)
        else:
            print("\nOMIM Download Instructions:")
            print("1. Go to https://www.omim.org/")
            print("2. Create an account")
            print("3. Request API access at https://www.omim.org/api")
            print("4. Once approved, download 'morbidmap.txt'")
            print(f"5. Save to: {omim_raw}")
            print("6. Run this script again")
    
    elif choice == '3':
        # PharmGKB instructions
        pharmgkb_raw = raw_dir / "relationships.tsv"
        pharmgkb_parsed = raw_dir / "pharmgkb_gene_disease.tsv"
        
        if pharmgkb_raw.exists():
            parse_pharmgkb_relationships(pharmgkb_raw, pharmgkb_parsed)
        else:
            download_pharmgkb_api()
    
    elif choice == '4':
        # DisGeNET instructions
        download_disgenet(raw_dir / "disgenet_gene_disease.tsv")
    
    elif choice.lower() == 'all':
        print("\n" + "="*60)
        print("COMPLETE DATA COLLECTION WORKFLOW")
        print("="*60)
        
        # Show all options
        download_clinvar_variant_summary(raw_dir / "variant_summary.txt")
        parse_omim_morbidmap(raw_dir / "morbidmap.txt", raw_dir / "omim_gene_disease.tsv")
        download_pharmgkb_api()
        download_disgenet(raw_dir / "disgenet_gene_disease.tsv")
        
        print("\n" + "="*60)
        print("MERGING DATA")
        print("="*60)
        print("\nAfter downloading all sources, run this script with option 'merge'")
    
    elif choice.lower() == 'merge':
        # Merge all available sources
        source_files = [
            raw_dir / "clinvar_gene_disease.tsv",
            raw_dir / "omim_gene_disease.tsv",
            raw_dir / "pharmgkb_gene_disease.tsv",
            raw_dir / "disgenet_gene_disease.tsv"
        ]
        
        final_output = data_dir / 'gene_disease_associations.tsv'
        merge_gene_disease_sources(source_files, final_output)
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"\nGene-disease associations ready at:")
        print(f"  {final_output}")
        print("\nYou can now run:")
        print("  python bin/gene_disease_network.py")
    
    else:
        print("\nInvalid option")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
