except KeyError as e:
    logging.error(f"Missing key in config: {e}")
    raise
except TypeError as e:
    logging.error(f"Invalid type in config: {e}")
    raise
except ValueError as e:
    logging.error(f"Invalid parameter value: {e}")
    raise
except Exception as e:
    logging.error(f"Unexpected error initializing models: {e}")
    raise

except FileNotFoundError:
    logging.error(f"Config not found at {config_path}. Please check file path.")
    raise
except KeyError as e:
    logging.error(f"Missing key in configuration file: {e}")
    raise
except yaml.YAMLError as e:
    logging.error(f"Error parsing YAML file: {e}")
    raise 