from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import os
import shutil
import yaml

app = FastAPI()



@app.get("/test/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    data = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    if not item_id in range(1, 6):
        return {"value": "sixty-nine"}
    return {"value": data[item_id]}

# GET METHODS
@app.get("/model-names")
def get_model_names():

    return {"response" : os.listdir("/code/training")}



# POST METHODS
class TrainParameters(BaseModel):
    weights : str
    cfg : str
    data : str
    hyp : str
    epochs : int
    batch_size : int
    img_size : int
    device : str
    sync_bn : bool
    workers : int
    name : str
@app.post("/train-model")
def train_model(train_params : TrainParameters):

    print("Training model...")

    cmd = "nohup python3 -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py"
    cmd += f" --workers {train_params.workers}"
    cmd += f" --device {train_params.device}"
    if train_params.sync_bn:
        cmd += f" --sync-bn"
    cmd += f" --batch-size {train_params.batch_size}"
    cmd += f" --data {train_params.data}"
    cmd += f" --img-size {train_params.img_size}"
    cmd += f" --cfg {train_params.cfg}"
    cmd += f" --weights {train_params.weights}"
    cmd += f" --name {train_params.name}"
    cmd += f" --hyp {train_params.hyp}"
    cmd += f" --epochs {train_params.epochs}"

    return {"response": cmd}

class CreateDirectoryOptions(BaseModel):
    model_name : str
    copy_old_model_dataset : bool
    old_model_name : Union[str, None] = None
    copy_old_model_weights : bool
    old_model_weights_path : Union[str, None] = None
@app.post("/create-new-model-directory")
def create_new_model_directory(options : CreateDirectoryOptions):

    # Create response string
    response = ""

    # Try create the new directory
    train_dir = os.path.join(os.getcwd(), "training")
    new_dir_name = os.path.join(train_dir, options.model_name)
    try:
        os.mkdir(new_dir_name)
        response += "Created model directory successfully"

    except FileExistsError:
        response += "Error: Model name already in use"
        return {"response" : response}
    except FileNotFoundError:
        response += "Error: Could not create model"
        return {"response" : response}
    
    # Make directories inside new model and copy some data from rf_intrusion_yolov7x_v3
    new_cfg_path = ""
    new_data_path = ""
    new_dataset_path = ""
    new_pretrained_weights_path = ""
    try:
        # Get folder paths to create
        new_cfg_path = os.path.join(new_dir_name, "cfg")
        new_data_path = os.path.join(new_dir_name, "data") 
        new_dataset_path = os.path.join(new_dir_name, "dataset")
        new_pretrained_weights_path = os.path.join(new_dir_name, "pre-trained-weights")

        # Make folders
        os.mkdir(new_cfg_path)
        os.mkdir(new_data_path)
        os.mkdir(new_dataset_path)
        os.mkdir(new_pretrained_weights_path)
        response += "\nCreated new directories"

        # Make alerts folder inside dataset folder
        new_alerts_path = os.path.join(new_dataset_path, "alerts")
        os.mkdir(new_alerts_path)

        # Get folder paths from rf_intrusion_yolov7x_v3 for folders to copy across
        v3_cfg_path = os.path.join(train_dir, os.path.join("rf_intrusion_yolov7x_v3", "cfg"))
        v3_pretrained_weights_path = os.path.join(
                train_dir,
                os.path.join("rf_intrusion_yolov7x_v3", "pre-trained-weights")
            )
        
        # Copy data across from cfg and pre-trained-weights folders
        try:
            copy_directory_contents(v3_cfg_path, new_cfg_path)
            copy_directory_contents(v3_pretrained_weights_path, new_pretrained_weights_path)

        except Exception:
            response += "\nError: could not copy standard data across to new model"
            return {"response" : response}
        response += "\nCopied cfg and pre-trained-weights folders across"
        
        # Copy some of the files and info from data folders across
        try:
            new_data_yaml_path = os.path.join(new_data_path, "intrusion.yaml")
            data  =  {
                "train" : f"/training/{options.model_name}/data/train.txt",
                "val" : f"/training/{options.model_name}/data/val.txt",  
                "test" : f"/training/{options.model_name}/data/test.txt",
                "nc" : 3,
                "names" : ['person', 'animal', 'vehicle']
            }
            with open(new_data_yaml_path, 'w') as yaml_file:
                yaml.dump(data, yaml_file, default_flow_style=False)
            yaml_file.close() 

        except Exception:
            response += "\nError: could not create intrusion.yaml file in data folder"
            return {"response" : response}
        
        # Try create train.txt and val.txt files
        train_txt_path = os.path.join(new_data_path, "train.txt")
        with open(train_txt_path, 'r') as train_txt:
            response += "\nCreated train.txt file"
        val_txt_path = os.path.join(new_data_path, "val.txt")
        with open(val_txt_path, 'r') as val_txt:
            response += "\nCreated new val.txt file"

    except FileNotFoundError:
        response += "\nError: could not create new directories"
        return {"response" : response}

    # Try copy the old dataset and data files across
    if options.copy_old_model_dataset:
        old_model_dir = os.path.join(train_dir, options.old_model_name)
        if os.path.exists(old_model_dir):
            # Try copy images from dataset across
            try:
                old_model_dataset_dir = os.path.join(
                                                old_model_dir, 
                                                os.path.join("dataset", "alerts")
                                            )
                copy_directory_contents(old_model_dataset_dir, new_dataset_path)
            except Exception:
                response += "\nError: could not copy dataset from old model across to new model"
                return {"response" : response}
            # Try copy data from train.txt and val.txt across
            try:
                old_model_data_dir = os.path.join(old_model_dir, "data")
                copy_text_file(old_model_data_dir, new_data_path, "train.txt")
                copy_text_file(old_model_data_dir, new_data_path, "val.txt")
            except Exception:
                response += "\nError: could not copy data from train.txt and val.txt across"
                {"response" : response}

    # Try copy across the weights from a previous model
    if options.copy_old_model_weights:
        try:
            if os.path.isfile(options.old_model_weights_path):
                copy_weights_path = os.path.join(
                        new_pretrained_weights_path, 
                        f"{options.old_model_name}.pt"
                    )
                shutil.copy2(options.old_model_weights_path, copy_weights_path)
        except AttributeError:
            response += "\nError: no path given to old model's weights"
            return {"response" : response}
        except Exception:
            response += "\nError: could not copy old weights across"
            return {"response" : response}

    # Return response
    response += "\nAll operations executed successfully"
    return {"response" : response}



# Utility functions
def copy_directory_contents(source_dir, destination_dir):

    try:
        # Iterate over all files and subdirectories in the source directory
        for item in os.listdir(source_dir):
            # Get the full path of the item
            source_item = os.path.join(source_dir, item)
            destination_item = os.path.join(destination_dir, item)

            # If the item is a file, copy it to the destination directory
            if os.path.isfile(source_item):
                shutil.copy2(source_item, destination_item)
            # If the item is a directory, recursively copy its contents
            elif os.path.isdir(source_item):
                copy_directory_contents(source_item, destination_item)

    except Exception as e:
        print(f"Error: {e}")
        raise Exception
    
def copy_text_file(source_dir, dest_dir, filename):

    try:
        # Check if the source file exists
        source_file_path = os.path.join(source_dir, filename)
        if not os.path.isfile(source_file_path):
            print(f"Source file '{filename}' does not exist in directory '{source_dir}'.")
            return

        # Read the contents of the source text file
        with open(source_file_path, 'r') as source_file:
            content = source_file.read()

        # Write the contents to the destination text file
        dest_file_path = os.path.join(dest_dir, filename)
        with open(dest_file_path, 'w') as dest_file:
            dest_file.write(content)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise Exception