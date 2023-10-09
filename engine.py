"""
will contain training functionality
"""
import tensorflow as tf
import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path

from tqdm.auto import tqdm    

import Tools.backboneUtils as backboneUtils
import Tools.loss_functions as loss_functions
import Tools.datasetTools as DatasetTools
import Tools.utilities as common_utils
from collections import defaultdict
import backbone_engine as backbone_engine



def train_first_stage(model: tf.keras.Model,
                    train_dataloader,
                    test_dataloader,
                    optimizer,
                    epochs=1,
                    from_checkpoint=None,
                    save_path="models",
                    save_as=f"featureEmbeddingBackBone",
                    save_frequency=1,
                    save_hard_frequency=50):
    
    # create a tag for the training
    log_tag = {
                    "model-name": model.name,
                    "epochs": epochs,
                    "resumed_from": None,
                    "train_size": len(train_dataloader),
                    "test_size": len(test_dataloader),
                    "tag_name": f"train_{model.name}_first_stage_{epochs}_epochs",
                    "per_epoch_metrics":{"train_loss": defaultdict(list),
                                            "test_results":defaultdict(list)
                                            },
                    "training_time": 0     
                }
    
    #if from_checkpoint is not None load the saved model
    if from_checkpoint is not None:
        pattern = f"*" if str(from_checkpoint)=="latest" else f"{from_checkpoint}*"
        model_name = common_utils.latest_file(Path(save_path), pattern=pattern)
        log_tag["resumed_from"] = str(model_name)
        model = common_utils.load_model(model_name)
        
        
        
    # 2. Create empty results dictionary
    print(f"[INFO] Training{model.name} for {epochs} epochs")

    from timeit import default_timer as timer
    start_time = timer()
    
    
    # 3. Loop over the epochs
    for  epoch in range(epochs):
        print(f"[INFO] Epoch {epoch+1}/{epochs}")
        _per_epoch_train_losses = backbone_engine.train_step(model=model,
                                                dataloader=train_dataloader,
                                                optimizer=optimizer)
        
        
        _per_epoch_test_results = backbone_engine.test_step(model=model,
                                                    dataloader=test_dataloader)
        # 6. Save model
        if (epoch+1) % save_frequency == 0:
            hard_tag = str(int((epoch+1)/save_hard_frequency) + 1)
            common_utils.save_model_weights(model=model,
                                            save_path=save_path,
                                            save_as=save_as,
                                            tag=str(hard_tag))
            
    
        # could be functionalized
        # 8.  Update logs dictionary
        for k,v in _per_epoch_train_losses.items():
            log_tag["per_epoch_metrics"]["train_loss"][k].append(v)
        for k,v in _per_epoch_test_results.items():
            log_tag["per_epoch_metrics"]["test_results"][k].append(v)
        
        end_time = timer()
        training_time = f"{(end_time-start_time)/3600:.2f} hours"
        log_tag["training_time"] = training_time

        # 7. Save results
        common_utils.save_logs(logs=log_tag,
                                save_path="logs",
                                save_as=f"{log_tag['tag_name']}.json")
                                
    return log_tag["per_epoch_metrics"] 
