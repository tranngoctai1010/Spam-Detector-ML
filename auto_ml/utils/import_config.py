import yaml

def load_import_config(config_file_path: str) -> tuple[list[str], list[dict]]:
    """
    Load configuration from a YAML file for dynamic module importing.

    Args:
        config_file_path (str): Path to the YAML configuration file.

    Returns:
        tuple[list[str], list[dict]]: A tuple containing two lists:
            - List of module paths as strings.
            - List of corresponding class configuration dictionaries.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
        yaml.YAMLError: If the YAML file is malformed or cannot be parsed.
    """
    if not isinstance(config_file_path, str):
        raise ValueError(f"{config_file_path} must be a tring.")
    try:
        with open(config_file_path, "r") as file:
            config = yaml.safe_load(file)

        if not config:
            raise ValueError(f"Configuration file is empty: {config_file_path}")

        module_paths, class_configs = zip(*[(path, conf) for path, conf in config.items()])
        return list(module_paths), list(class_configs)

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_file_path}: {str(e)}")