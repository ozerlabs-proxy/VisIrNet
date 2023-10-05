import json
import numpy as np
from pathlib import Path
import tensorflow as tf

def tensor_has_nan(some_tensor):
        """
        check if the tensor has nans
        """

        
        return tf.math.is_nan(some_tensor).numpy().any()


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
                

def save_model_weights(model, 
                       save_path, 
                       save_as,
                       tag = None,
                       ):
        """
        save model weights
        """
        tag = "" if tag is None else tag
        try:
                print(f"[INFO] Saving model at {save_path}")
                save_path = Path(save_path)
                save_path.mkdir(parents=True, exist_ok=True)
                save_path = str(save_path / f"{save_as}_{tag}.h5")
                model.save(save_path)
        except Exception as e:
                print(f"[ERROR] model couldn't be saved at {save_path}")
                
def save_logs(logs,save_path,save_as):
        """
                save logs to json file given the logs (metrics) dictionary
        """
        save_path = Path(save_path)
        try:
                print(f"[INFO] Saving logs at {save_path}")
                save_path.mkdir(parents=True, exist_ok=True)
                save_path = save_path / f"{save_as}"
                assert save_path.suffix == '.json', "file name must have .json extension"
                save_path = str(save_path)
                with open(save_path, 'w') as fp:
                        json.dump(logs, fp,default=np_encoder)
        except Exception as e:
                print(f"[ERROR] model couldn't be saved at {save_path}")
        