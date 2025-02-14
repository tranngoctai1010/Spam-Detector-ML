except FileNotFoundError:
    logging.error(f"Config not found at {config_path}. Please check file path.")
    raise
except KeyError as e:
    logging.error(f"Missing key in configuration file: {e}")
    raise
except yaml.YAMLError as e:
    logging.error(f"Error parsing YAML file: {e}")
    raise
