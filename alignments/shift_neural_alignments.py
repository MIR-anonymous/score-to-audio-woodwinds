from common_variables_functions import *

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#############################################################

NEURALSHIFT= -0.010524837018133226
STSHIFT= 0.004101957027131675
CHROMASHIFT= -0.006863484810622822

#############################################################

def shift_alignments(rep: str, performer: str, fantasia: int):
    """
    Apply a global onset time shift to alignment CSV files and save updated versions.

    This function loads an alignment file corresponding to a given representation
    (chroma, st, or neural), applies a fixed temporal correction to the "Onset (ms)"
    column, and writes the shifted result to an output directory.

    The shift values are defined as global constants (in seconds) and converted to
    milliseconds before being applied.

    Args:
        rep : Representation type. Must be one of:
            - 'chroma' : chroma-based
            - 'st'     : Spectral Template-based
            - 'neural' : Neural MPE-based

        performer: Name/identifier of the performer. Used to locate input files.

        fantasia: Fantasia number (1-12), used to select the corresponding CSV file.

    Input:
        CSV file located in:
            ALIGNMENTS_{REP}/{performer}/{rep}_f{fantasia}.csv

        The CSV must contain a column:
            "Onset (ms)"

    Processing:
        The onset times are corrected by subtracting a global shift:
            - CHROMASHIFT for chroma representations
            - STSHIFT for st representations
            - NEURALSHIFT for neural representations

        Shift values are originally in seconds and converted to milliseconds.

    Output:
        A shifted CSV file is saved to:
            output/shifted/{rep}/{performer}/{rep}-shifted-f{fantasia}.csv
    """
    if rep == 'chroma' :
        input_file = f"{ALIGNMENTS_CHROMA}{performer}/chroma_f{fantasia}.csv" 
        output_dir = f"output/shifted/chroma/{performer}" 
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.read_csv(input_file)
        df["Onset (ms)"] = df["Onset (ms)"] - CHROMASHIFT*1000
        df.to_csv(f'{output_dir}/chroma-shifted-f{fantasia}.csv', index=False)

    elif rep == 'st' :
        input_file = f"{ALIGNMENTS_ST}{performer}/st_f{fantasia}.csv" 
        output_dir = f"output/shifted/st/{performer}" 
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.read_csv(input_file)
        df["Onset (ms)"] = df["Onset (ms)"] - STSHIFT*1000
        df.to_csv(f'{output_dir}/st-shifted-f{fantasia}.csv', index=False)
    
    elif rep == 'neural' :
        input_file = f"{ALIGNMENTS_NEURAL}{performer}/neural_f{fantasia}.csv" 
        output_dir = f"output/shifted/neural/{performer}" 
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.read_csv(input_file)
        df["Onset (ms)"] = df["Onset (ms)"] - NEURALSHIFT*1000
        df.to_csv(f'{output_dir}/neural-shifted-f{fantasia}.csv', index=False)

if __name__== '__main__':
    for performer in ALL:
        for fantasia in range(1,13):
            for rep in ['chroma', 'st', 'neural']:
                shift_alignments(rep, performer, fantasia)