from collections import namedtuple
import json
from pathlib import Path

class ConfigurationParser():
    
    @staticmethod
    def getConfigurations(configs_path = "configs", config_file = "default_config.json"):
        """
            read configurations from a json file and return a namedtuple
        """

        config_path = Path(configs_path) / f'{config_file}'
        
        assert config_path.exists() , "Config file does not exist"
        assert config_file.endswith(".json") , "Config file should be a json file"
        
        try:
            json_configs = json.load(open(config_path))
        except Exception as e:
            print(f"[ERROR] reading configs from {config_path}")
            exit(1)
        
        ConfigItem = namedtuple("Configs", json_configs.keys())
        Configs = ConfigItem(*json_configs.values())
        
        # tweak configurations
        Configs = Configs._replace(REGRESSION_INPUT_SHAPE = [*Configs.RGB_INPUTS_SHAPE[:2], Configs.OUTPUT_CHANNELS_PER_BLOCK*2])
        Configs = Configs._replace(B_save_path = f"models/{Configs.dataset}")
        Configs = Configs._replace(R_save_path = f"models/{Configs.dataset}")
        #
        
        assert Configs.REGRESSION_INPUT_SHAPE != None, "REGRESSION_INPUT_SHAPE should not be None"
        
        return Configs
    
    @staticmethod
    def printConfigurations(configs):
        print("*"*25, f"Configurations", "*"*25)
        for k, v in configs._asdict().items():
            print(f"\t {k}: {v}")
            
        print("*"*65)