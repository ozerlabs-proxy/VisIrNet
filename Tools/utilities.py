import json
import numpy as np
from pathlib import Path

# dumping json needs to convert numpy types to python types
def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

# save coco format to json file 
def save_json(data, 
          save_dir:str,
          file_name:str):
        """
        Save the json file
        """

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / file_name
        
        assert save_path.suffix == '.json', "file name must have .json extension"
        
        print("[INFO] saving json file to {}".format(save_path))
        try:
                with open(save_path, 'w') as fp:
                        json.dump(data, fp,default=np_encoder)
        except Exception as e:
                print("[ERROR] error saving json file...")
                print(e)
                return
        print(f"[INFO] file saved at {save_dir}.")
                
