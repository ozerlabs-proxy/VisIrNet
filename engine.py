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
import regression_head_engine as regression_head_engine


""" train first stage (feature embedding backbone)"""
def train_first_stage(model: tf.keras.Model,
                    train_dataloader,
                    test_dataloader,
                    dataset_name,
                    optimizer,
                    epochs=1,
                    from_checkpoint=None,
                    save_path=None,
                    save_as=f"featureEmbeddingBackBone",
                    save_frequency=1,
                    save_hard_frequency=50,
                    loss_function="mse_pixel",
                    uuid=""):
    assert save_path is not None, "save_path is None"
    
    save_path = save_path+ f"/{loss_function}"
    # create a tag for the training
    log_tag = {
                    "model-name": model.name,
                    "epochs": epochs,
                    "loss_function": loss_function,
                    "resumed_from": None,
                    "train_size": len(train_dataloader),
                    "test_size": len(test_dataloader),
                    "tag_name": f"{dataset_name}-{loss_function}-{model.name}-1-{epochs}-{uuid}",
                    "per_epoch_metrics":{"train_loss": defaultdict(list),
                                            "test_results":defaultdict(list)
                                            },
                    "training_time": 0     
                }
    
    #if from_checkpoint is not None load the saved model
    if from_checkpoint is not None:
        pattern = f"*{loss_function}-{model.name}*" if str(from_checkpoint)=="latest" else f"{from_checkpoint}*"
        model_name = common_utils.latest_file(Path(save_path), pattern=pattern)
        log_tag["resumed_from"] = str(model_name)
        model = common_utils.load_model(model_name)
        
        
        
    # 2. Create empty results dictionary
    print(f"[INFO] Training {model.name} for {epochs} epochs")

    from timeit import default_timer as timer
    start_time = timer()
    
    # 3. Create summary writer
    _train_summary_writer = common_utils.get_summary_writter(log_dir = "logs/tensorboard",
                                                    log_id=f"{dataset_name}-b-{loss_function}-{str(uuid)}",
                                                    suffix="1-train")
    _test_summary_writer = common_utils.get_summary_writter(log_dir = "logs/tensorboard",
                                                    log_id=f"{dataset_name}-b-{loss_function}-{str(uuid)}",
                                                    suffix="1-test")

    
    # 4. Loop over the epochs
    
    # compile the model with the optimizer
    model.compile(optimizer=optimizer) 
    
    for  epoch in range(epochs):
        print(f"[INFO] Epoch {epoch+1}/{epochs}")
        model , _per_epoch_train_losses , train_log = backbone_engine.train_step(model = model,
                                                                        dataloader = train_dataloader,
                                                                        optimizer = optimizer,
                                                                        loss_function = loss_function)
        
        
        _per_epoch_test_results , test_log = backbone_engine.test_step(model=model,
                                                            dataloader=test_dataloader,
                                                            loss_function=loss_function)
        
        print(f"[train_loss] : {train_log}")
        print(f"[test_loss] : {test_log}")
        
        # 6. Save model
        if (epoch+1) % save_frequency == 0:
            hard_tag = str(int((epoch+1)/save_hard_frequency) + 1)
            common_utils.save_model_weights(model = model,
                                            save_path = save_path,
                                            save_as=f"{log_tag['tag_name']}",
                                            tag = str(hard_tag))
            
    
        # could be functionalized
        # 8.  Update logs dictionary
        for k,v in _per_epoch_train_losses.items():
            log_tag["per_epoch_metrics"]["train_loss"][k].append(v)
        for k,v in _per_epoch_test_results.items():
            log_tag["per_epoch_metrics"]["test_results"][k].append(v)
        
        end_time = timer()
        training_time = (end_time-start_time)/3600
        log_tag["training_time"] = f"{training_time:.2f} hours"
        
        common_utils.tb_write_summary(_summary_writer = _train_summary_writer, 
                                        epoch = epoch ,
                                        logs = _per_epoch_train_losses)

        common_utils.tb_write_summary(_summary_writer = _test_summary_writer, 
                                        epoch = epoch ,
                                        logs = _per_epoch_test_results )

        common_utils.tb_write_summary(_summary_writer = _train_summary_writer, 
                                        epoch = epoch ,
                                        logs = {"training_time": tf.cast(training_time, tf.float32).numpy()})

        # 7. Save results
        common_utils.save_logs(logs=log_tag,
                                save_path=f"logs/{dataset_name}",
                                save_as=f"{log_tag['tag_name']}.json")
                                
    return model, log_tag["per_epoch_metrics"] 

""" train second stage (regression head)"""

def train_second_stage(model: tf.keras.Model,
                        featureEmbeddingBackBone,                        
                        train_dataloader ,
                        test_dataloader,
                        dataset_name,
                        optimizer,
                        epochs=1,
                        from_checkpoint=None,
                        save_path=None,
                        save_as=f"regressionHead",
                        save_frequency=1,
                        save_hard_frequency=50,
                        predicting_homography=False,
                        backbone_loss_function="mse_pixel",
                        loss_function_to_use="l2_homography_loss",
                        uuid=""):
    assert save_path is not None, "save_path is None"
    save_path_backbone = save_path+ f"/{backbone_loss_function}"
    save_path = save_path+ f"/{loss_function_to_use}"
    homography_based = "homography" if predicting_homography else "corners"

    # create a tag for the training
    log_tag = {
                    "model-name": model.name,
                    "epochs": epochs,
                    "resumed_from": None,
                    "train_size": len(train_dataloader),
                    "test_size": len(test_dataloader),
                    "featureEmbeddingBackBone": None,
                    "predicting_homography": predicting_homography,
                    "tag_name": f"{dataset_name}-{model.name}-{backbone_loss_function}-{loss_function_to_use}-{homography_based}-2-{epochs}-{uuid}",
                    "per_epoch_metrics":{
                        "backbone_train_loss": defaultdict(list),
                        "backbone_test_results":defaultdict(list),
                        "train_loss": defaultdict(list),
                        "test_results":defaultdict(list)
                        },
                    "training_time": 0     
                }
    backBone = None
    #if from_checkpoint is not None load the saved model
    if from_checkpoint is not None:
        pattern = f"*{model.name}-{backbone_loss_function}-{loss_function_to_use}*" if str(from_checkpoint)=="latest" else f"*{from_checkpoint}*"
        model_name = common_utils.latest_file(Path(save_path), pattern=pattern)
        log_tag["resumed_from"] = str(model_name)
        model = common_utils.load_model(model_name)
        
    # load the feature embedding backbone
    if featureEmbeddingBackBone is not None:
        pattern = f"*{backbone_loss_function}-featureEmbeddingBackbone*" if str(featureEmbeddingBackBone)=="latest" else f"*{featureEmbeddingBackBone}*"
        model_name = common_utils.latest_file(Path(save_path_backbone), pattern=pattern)
        log_tag["featureEmbeddingBackbone"] = str(model_name)
        backBone = common_utils.load_model(model_name)
    
        
    # 2. Create empty results dictionary
    print(f"[INFO] Training {model.name} for {epochs} epochs")

    from timeit import default_timer as timer
    start_time = timer()
    
    # 3. Create summary writer
    _train_summary_writer = common_utils.get_summary_writter(log_dir = "logs/tensorboard",
                                                    log_id=f"{dataset_name}-regression_head-{str(uuid)}",
                                                    suffix="2-train")
    _test_summary_writer = common_utils.get_summary_writter(log_dir = "logs/tensorboard",
                                                    log_id=f"{dataset_name}-regression_head-{str(uuid)}",
                                                    suffix="2-test")
    # 3. Loop over the epochs
    
    # compile the model with the optimizer
    model.compile(optimizer=optimizer) 
    
    for  epoch in range(epochs):
        print(f"[INFO] Epoch {epoch+1}/{epochs}")
        model, _per_epoch_train_losses , train_log = regression_head_engine.train_step(model=model,
                                                                                            backBone=backBone,
                                                                                            dataloader=train_dataloader,
                                                                                            optimizer=optimizer,
                                                                                            predicting_homography = predicting_homography,
                                                                                            backbone_loss_function = backbone_loss_function,
                                                                                            loss_function_to_use = loss_function_to_use)
        
        
        _per_epoch_test_results ,test_log = regression_head_engine.test_step(model=model,
                                                                            backBone=backBone,
                                                                            dataloader=test_dataloader,
                                                                            predicting_homography = predicting_homography,
                                                                            backbone_loss_function = backbone_loss_function,
                                                                            loss_function_to_use = loss_function_to_use
                                                                            )
        # 6. Save model
        if (epoch+1) % save_frequency == 0:
            hard_tag = str(int((epoch+1)/save_hard_frequency) + 1)
            common_utils.save_model_weights(model=model,
                                            save_path=save_path,
                                            save_as=f"{log_tag['tag_name']}",
                                            tag=str(hard_tag))
            
    
        # could be functionalized
        # 8.  Update logs dictionary
        for step, losses in _per_epoch_train_losses.items():
            step = "" if str(step) != "backbone" else "backbone_"
            for key, value in losses.items():
                log_tag["per_epoch_metrics"][f"{step}train_loss"][key].append(value)
                
        for step, results in _per_epoch_test_results.items():
            step = "" if str(step) != "backbone" else "backbone_"
            for key, value in results.items():
                log_tag["per_epoch_metrics"][f"{step}test_results"][key].append(value)
        
        end_time = timer()
        training_time = (end_time-start_time)/3600
        log_tag["training_time"] = f"{training_time:.2f} hours"
        
        common_utils.tb_write_summary(_summary_writer = _train_summary_writer, 
                                        epoch = epoch ,
                                        logs = _per_epoch_train_losses["regression_head"]
                                        )
        common_utils.tb_write_summary(_summary_writer = _test_summary_writer, 
                                        epoch = epoch ,
                                        logs = _per_epoch_test_results["regression_head"] 
                                        )
        common_utils.tb_write_summary(_summary_writer = _train_summary_writer, 
                                        epoch = epoch ,
                                        logs = {"training_time": training_time})


        # 7. Save results
        common_utils.save_logs(logs=log_tag,
                                save_path=f"logs/{dataset_name}",
                                save_as=f"{log_tag['tag_name']}.json")
        
        print(f"[TRAIN] : {train_log}")
        print(f"[TEST] : {test_log}")
                                
    return model, log_tag["per_epoch_metrics"] 
