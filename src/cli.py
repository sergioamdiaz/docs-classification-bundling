#*******************************************************************************
# IMPORTS:
#*******************************************************************************

import argparse
from pathlib import Path

try:
    from src.data import load_config
    print("Data module imported correctly \n")
    
    from src.pipeline import main_function
    print("Pipeline module imported correctly \n")
except ImportError as e:
    print(f"Import Error: {e} \n")

#*******************************************************************************
# BLOCK MAIN:
#*******************************************************************************

if __name__ == "__main__":
    """ To run the code, activate the virtual environment and execute the following 
    command in the CLI, providing the path to your config.yaml file:
    >>> python -m modules.kmeans_module --config config/config.yaml  
    Assumptions: 
        - The config.yaml file is located in the config/ folder at the project root.
        - The command is executed from the root of the project.
    """
    # --------------------------------------------------------------------------
    # Load configuration:
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="K-means clustering for document classification.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.") # argparse automatically converts the type to Path.
    args = parser.parse_args() # Here is when the scripts reads from the command line.
    config_path = args.config # The name of the attribute "config" is taken from the flag name in add_argument().
    if not config_path.exists():
        raise FileNotFoundError('\n' + '-'*70 + f"\nConfig file was not found at: {config_path}\n" + '-'*70)
    config: dict = load_config(config_path) # Loads the content of the YAML file as a dict.
    
    root_dir = Path(__file__).resolve().parent.parent # Assumes the script is in modules/ folder, and modules/ in the root of the project.
    
    # --------------------------------------------------------------------------
    # Extract the main paths from the config dict (paths should be relative to the root of the project):
    # --------------------------------------------------------------------------
    data_dir = root_dir / config["paths"]["data_dir"]
    if not data_dir.exists():
        raise FileNotFoundError('\n' + '-'*70 + f"\nThe data directory was not found at: {data_dir}\n" + '-'*70)
    descriptions_file = root_dir / config["paths"]["descriptions_file"]
    if not descriptions_file.exists():
        raise FileNotFoundError('\n' + '-'*70 + f"\nThe descriptions file was not found at: {descriptions_file}\n" + '-'*70)
    output_dir = root_dir / config["paths"]["output_dir"]
    types = config["types"] # List of doc-types to build each master PDF.
    if not types or not isinstance(types, list):
        raise ValueError('\n' + '-'*70 + f"\nThe 'types' key in the config file must be a non-empty list of document types.\n" + '-'*70)

    main_function(data_dir, descriptions_file, output_dir, types)