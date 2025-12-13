"""
Download and prepare Protein-Protein Interaction (PPI) data.

This script integrates five publicly available PPI databases:
1. HPRD (Human Protein Reference Database)
2. BioGrid
3. DIP (Database of Interacting Proteins)
4. IntAct
5. MINT (Molecular Interaction Database)

Reference:
Integrated PPI network with 104,522 interactions among 14,212 proteins.
"""

import pandas as pd
import requests
from pathlib import Path
from io import StringIO
import gzip


def download_biogrid(output_path, organism_id=9606, exclude_non_physical=False):
    """
    Download BioGrid PPI data.
    
    BioGrid is freely available and integrates interactions from multiple sources.
    
    Parameters:
    -----------
    output_path : Path
        Where to save the data
    organism_id : int
        NCBI taxonomy ID (9606 for human)
    exclude_non_physical : bool
        If True, only include physical interactions
    """
    print("\n" + "="*60)
    print("DOWNLOADING BIOGRID PPI DATA")
    print("="*60)
    
    # BioGrid download URL
    url = f"https://downloads.thebiogrid.org/Download/BioGRID/Release-Current/BIOGRID-ALL-LATEST.tab.zip"
    
    print(f"\nDownloading from: {url}")
    print("This may take a while...")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(output_path.parent / "BIOGRID.zip", 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='')
        
        print(f"\n\nDownloaded to: {output_path.parent / 'BIOGRID.zip'}")
        print("Please extract the zip file")
        print("Look for file matching pattern: BIOGRID-ALL-*-human.tab")
        print(f"Then save as: {output_path}")
        
        return output_path.parent / "BIOGRID.zip"
        
    except Exception as e:
        print(f"Error downloading: {e}")
        print("\nManual download:")
        print(f"1. Visit: https://thebiogrid.org/download.php")
        print("2. Download latest human PPI file")
        print(f"3. Save to: {output_path}")
        return None


def parse_biogrid_file(filepath, output_path):
    """
    Parse BioGrid interaction file using **official gene symbols**.

    BioGrid columns we care about:
    - "Official Symbol Interactor A"
    - "Official Symbol Interactor B"
    Fallback to Entrez IDs if symbols are missing.
    """
    print("\n" + "="*60)
    print("PARSING BIOGRID DATA")
    print("="*60)

    if not filepath.exists():
        print(f"File not found: {filepath}")
        return None

    print(f"Reading {filepath}")

    # Read with header so we can pick the correct columns
    df_raw = pd.read_csv(
        filepath,
        sep='\t',
        low_memory=False,
        dtype=str
    )

    # Preferred symbol columns
    sym_a_col = 'Official Symbol Interactor A'
    sym_b_col = 'Official Symbol Interactor B'
    # Fallback Entrez ID columns
    ent_a_col = 'Entrez Gene Interactor A'
    ent_b_col = 'Entrez Gene Interactor B'

    if sym_a_col in df_raw.columns and sym_b_col in df_raw.columns:
        df = df_raw[[sym_a_col, sym_b_col]].rename(columns={sym_a_col: 'protein_a', sym_b_col: 'protein_b'})
    elif ent_a_col in df_raw.columns and ent_b_col in df_raw.columns:
        df = df_raw[[ent_a_col, ent_b_col]].rename(columns={ent_a_col: 'protein_a', ent_b_col: 'protein_b'})
    else:
        raise ValueError("BioGrid file missing expected symbol or Entrez columns")

    # Drop self-interactions and empties
    df = df.dropna(subset=['protein_a', 'protein_b'])
    df = df[df['protein_a'] != df['protein_b']]

    # Add source
    df['source'] = 'BioGrid'

    print(f"\nParsed {len(df)} interactions")
    unique_proteins = pd.unique(df[['protein_a', 'protein_b']].values.ravel())
    print(f"Unique proteins: {len(unique_proteins)}")

    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved to: {output_path}")

    return df


def download_string_db(output_path, confidence_threshold=0.7):
    """
    Download STRING database PPI data (public alternative to multiple databases).
    
    STRING integrates interactions from multiple sources and is publicly available.
    
    Parameters:
    -----------
    output_path : Path
        Where to save the data
    confidence_threshold : float
        Minimum confidence score (0-1)
    """
    print("\n" + "="*60)
    print("DOWNLOADING STRING DATABASE PPI")
    print("="*60)
    
    print("\nSTRING database provides high-quality PPI data.")
    print("Visit: https://string-db.org/cgi/download.pl")
    print("\nOptions:")
    print("1. Download protein.actions.v*.txt.gz (for high confidence interactions)")
    print("2. Download protein.links.v*.txt.gz (for all interactions with scores)")
    print(f"\nSave to: {output_path}")
    
    return None


def parse_string_db_file(filepath, output_path, confidence_threshold=0.7):
    """
    Parse STRING database file.
    
    Parameters:
    -----------
    filepath : Path
        Path to STRING interaction file
    output_path : Path
        Where to save parsed interactions
    confidence_threshold : float
        Minimum confidence score
    """
    print("\n" + "="*60)
    print("PARSING STRING DATABASE")
    print("="*60)
    
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return None
    
    print(f"Reading {filepath}")
    
    # Open gzip file if needed
    if str(filepath).endswith('.gz'):
        open_func = gzip.open
    else:
        open_func = open
    
    interactions = []
    
    with open_func(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        # Skip header
        next(f)
        
        for i, line in enumerate(f):
            parts = line.strip().split()
            
            if len(parts) >= 3:
                protein_a = parts[0].strip()
                protein_b = parts[1].strip()
                score = float(parts[2]) if len(parts) > 2 else 1.0
                
                if score >= confidence_threshold:
                    interactions.append({
                        'protein_a': protein_a,
                        'protein_b': protein_b,
                        'score': score,
                        'source': 'STRING'
                    })
            
            if (i + 1) % 100000 == 0:
                print(f"Processed {i + 1} lines...")
    
    df = pd.DataFrame(interactions)
    print(f"\nParsed {len(df)} interactions (score >= {confidence_threshold})")
    print(f"Unique proteins: {df[['protein_a', 'protein_b']].values.ravel().nunique()}")
    
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved to: {output_path}")
    
    return df


def download_intact(output_path):
    """
    Download IntAct PPI data.
    
    IntAct is freely available and provides curated molecular interactions.
    """
    print("\n" + "="*60)
    print("DOWNLOADING INTACT")
    print("="*60)
    
    print("\nIntAct provides freely available PPI data.")
    print("Visit: https://www.ebi.ac.uk/intact/downloads")
    print("\nOptions:")
    print("1. Download 'intact.txt' (PSI-MI TAB format)")
    print("2. Filter for Homo sapiens")
    print(f"\nSave to: {output_path}")
    
    return None


def download_dip(output_path):
    """
    Download DIP (Database of Interacting Proteins).
    """
    print("\n" + "="*60)
    print("DOWNLOADING DIP")
    print("="*60)
    
    print("\nDIP provides high-quality, manually curated interactions.")
    print("Visit: https://dip.doe-mbi.ucla.edu/dip/Download.cgi")
    print("\nDownload human interactions in tab format")
    print(f"Save to: {output_path}")
    
    return None


def download_mint(output_path):
    """
    Download MINT (Molecular Interaction Database).
    
    Note: MINT has been integrated into IntAct.
    """
    print("\n" + "="*60)
    print("DOWNLOADING MINT")
    print("="*60)
    
    print("\nMINT has been integrated into IntAct.")
    print("Use IntAct download instead (option 4)")
    
    return None


def create_example_ppi_data(output_path):
    """
    Create example PPI data for testing.
    
    Parameters:
    -----------
    output_path : Path
        Where to save example data
    """
    print("\nCreating example PPI data...")
    
    # Create a small example network
    data = {
        'protein_a': [
            'BRCA1', 'BRCA1', 'BRCA2', 'BRCA2', 'TP53', 'TP53',
            'APP', 'APP', 'APOE', 'APOE',
            'SNCA', 'SNCA', 'LRRK2',
        ],
        'protein_b': [
            'TP53', 'RAD51', 'TP53', 'PALB2', 'MDM2', 'BAX',
            'APOE', 'APP', 'APOB', 'APP',
            'LRRK2', 'PINK1', 'PINK1',
        ],
        'source': ['Example'] * 13
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved example data to: {output_path}")
    print(f"Proteins: {df[['protein_a', 'protein_b']].values.ravel().nunique()}")
    print(f"Interactions: {len(df)}")
    
    return df


def merge_ppi_sources(source_files, output_path):
    """
    Merge PPI data from multiple sources.
    
    Parameters:
    -----------
    source_files : list of Path
        List of PPI files from different sources
    output_path : Path
        Where to save merged data
    """
    print("\n" + "="*60)
    print("MERGING PPI SOURCES")
    print("="*60)
    
    all_data = []
    
    for filepath in source_files:
        if filepath.exists():
            df = pd.read_csv(filepath, sep='\t')
            all_data.append(df)
            print(f"Loaded {len(df)} interactions from {filepath.name}")
    
    if not all_data:
        print("No PPI files found to merge")
        return None
    
    # Combine all sources
    merged = pd.concat(all_data, ignore_index=True)
    
    print(f"\nBefore deduplication: {len(merged)} interactions")
    
    # Remove duplicates (considering both directions as the same)
    merged['pair'] = merged.apply(
        lambda row: tuple(sorted([row['protein_a'], row['protein_b']])),
        axis=1
    )
    merged = merged.drop_duplicates(subset=['pair'])
    merged = merged.drop(columns=['pair'])
    
    print(f"After deduplication: {len(merged)} interactions")
    
    # Get unique proteins
    proteins = set(merged['protein_a']) | set(merged['protein_b'])
    print(f"Unique proteins: {len(proteins)}")
    
    # Save merged data
    merged.to_csv(output_path, sep='\t', index=False)
    print(f"\nMerged PPI data saved to: {output_path}")
    
    # Statistics by source
    if 'source' in merged.columns:
        print("\nInteractions by source:")
        for source in merged['source'].unique():
            count = len(merged[merged['source'] == source])
            print(f"  {source}: {count}")
    
    return merged


def main():
    """Main execution."""
    
    print("="*60)
    print("PROTEIN-PROTEIN INTERACTION (PPI) DATA DOWNLOAD")
    print("="*60)
    
    # Setup directories
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'replication'
    raw_dir = data_dir / 'raw_ppi'
    raw_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nData will be saved to: {raw_dir}")
    
    # Options
    print("\n" + "="*60)
    print("PPI DATABASE OPTIONS")
    print("="*60)
    print("\nAvailable sources (integrated databases):")
    print("1. BioGrid - FREE, comprehensive, auto-downloadable")
    print("2. STRING - FREE, high-quality, curated")
    print("3. IntAct - FREE, manually curated (manual download)")
    print("4. DIP - Curated, high precision (manual download)")
    print("5. MINT - Integrated into IntAct (use option 3)")
    print("6. Example data - For testing")
    print("7. Show instructions - For manual downloads")
    
    choice = input("\nSelect option (1-7): ").strip()
    
    if choice == '1':
        # BioGrid
        print("\nAttempting to download BioGrid...")
        biogrid_raw = raw_dir / "BIOGRID.zip"
        biogrid_parsed = raw_dir / "biogrid_ppi.tsv"
        
        biogrid_extracted = raw_dir / "BIOGRID-ALL-human.tab"
        
        if biogrid_extracted.exists():
            parse_biogrid_file(biogrid_extracted, biogrid_parsed)
        else:
            print("\nBioGrid download:")
            print("1. Visit: https://thebiogrid.org/download.php")
            print("2. Download human PPI file (BIOGRID-ALL-*-human.tab)")
            print(f"3. Save to: {biogrid_extracted}")
            print("4. Run this script again")
    
    elif choice == '2':
        # STRING
        print("\nSTRING database download:")
        print("1. Visit: https://string-db.org/cgi/download.pl")
        print("2. Download human protein.links file")
        string_file = raw_dir / "string_interactions.txt.gz"
        string_parsed = raw_dir / "string_ppi.tsv"
        
        if string_file.exists():
            parse_string_db_file(string_file, string_parsed)
        else:
            print(f"3. Save to: {string_file}")
            print("4. Run this script again")
    
    elif choice == '3':
        # IntAct
        download_intact(raw_dir / "intact_ppi.tsv")
    
    elif choice == '4':
        # DIP
        download_dip(raw_dir / "dip_ppi.tsv")
    
    elif choice == '5':
        # MINT (now IntAct)
        print("\nMINT has been integrated into IntAct.")
        download_intact(raw_dir / "intact_ppi.tsv")
    
    elif choice == '6':
        # Example data
        print("\nCreating example PPI network for testing...")
        example_file = raw_dir / "example_ppi.tsv"
        create_example_ppi_data(example_file)
        
        # Copy to final location
        final_output = data_dir / 'ppi_interactions.tsv'
        import shutil
        shutil.copy(example_file, final_output)
        print(f"\nExample PPI data ready at: {final_output}")
        print("Run: python bin/ppi_disease_network.py")
    
    elif choice == '7':
        # Show full instructions
        print("\n" + "="*60)
        print("COMPLETE PPI DATA COLLECTION WORKFLOW")
        print("="*60)
        
        print("\n1. BioGrid (Recommended):")
        print("   - Most comprehensive human PPI data")
        print("   - Visit: https://thebiogrid.org/download.php")
        print("   - Download: BIOGRID-ALL-*-human.tab")
        
        print("\n2. STRING Database:")
        print("   - High-quality, integrated predictions and experiments")
        print("   - Visit: https://string-db.org/cgi/download.pl")
        print("   - Download: protein.links.v*.txt.gz (human)")
        
        print("\n3. IntAct:")
        print("   - Manually curated, high precision")
        print("   - Visit: https://www.ebi.ac.uk/intact/downloads")
        print("   - Download: intact.txt")
        
        print("\n4. DIP:")
        print("   - High-quality, curated interactions")
        print("   - Visit: https://dip.doe-mbi.ucla.edu/dip/Download.cgi")
        
        print("\n5. After downloading:")
        print(f"   - Extract/prepare files")
        print(f"   - Save to {raw_dir}")
        print("   - Run: python bin/ppi_disease_network.py")
    
    else:
        print("Invalid option")


if __name__ == "__main__":
    main()
