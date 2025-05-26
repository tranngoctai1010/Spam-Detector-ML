from ..base import BaseFileLoader
import yaml

class YAMLFileLoader(BaseFileLoader):
    @classmethod
    def load_yaml_file(cls, config_path):
        try:
            cls._validate_load_method_params(config)
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML format: Expected a dictionary")
            return config
        except Exception:
            raise FileNotFoundError(f"Config file {config_path} not found")
        except Exception as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")
