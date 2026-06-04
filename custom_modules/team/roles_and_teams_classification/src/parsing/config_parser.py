import yaml
from ast import literal_eval
from typing import Dict


def load_config(config_path: str) -> Dict:
    """
    Loads and parses configuration data from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the configuration YAML file.

    Returns
    -------
    dict
        Dictionary of with parsed configuration fields.
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    
    matches_list = data.get('matches_ids', [])
    
    transformed_matches = {}
    keys_to_parse = ["folder", "sngs_inside", "remapping", "sngs_times", "pure_colors", "model_params_id"]
    
    for match in matches_list:
        match_id = match.get('id')
        
        if match_id is None:
            continue
            
        for key in keys_to_parse:
            if key in match and isinstance(match[key], str):
                try:
                    match[key] = literal_eval(match[key])
                except (ValueError, SyntaxError):
                    pass

        transformed_matches[match_id] = match
    return transformed_matches


def find_by_sngs(config_data: Dict[int, Dict], sngs: str) -> Dict:
    """
    Finds remapping dictionary, model id for ClearML and gametime in configuration file.

    Parameters
    ----------
    config_data : dict[int, dict]
        Сonfigurations with SNGSs slit and etc.
    sngs : str
        Target SNGS key in format 'SNGS-<NUM>'.

    Returns
    -------
    dict
        Info with remapping, model params ID, and time, or empty if not found.
    """
    for match_id, match_info in config_data.items():    
        sngs_list = match_info.get('sngs_inside', [])
        sngs_times = match_info.get('sngs_times', [])
        if (sngs in sngs_list) and (sngs in sngs_times):
            main_info = {
                'remapping': match_info.get('remapping'),
                'model_params_id': match_info.get('model_params_id'),
                'time_now': sngs_times[sngs]
            }
            return main_info
    return {}