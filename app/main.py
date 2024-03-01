from typing import Union, List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
import os
import shutil
import yaml
import logging
import subprocess
import asyncio

app = FastAPI()
logging.basicConfig(filename="app.log", 
                    filemode='w', 
                    format='%(name)s - %(levelname)s - %(message)s')

# GET METHODS

@app.get("/test/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    data = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    if not item_id in range(1, 6):
        return {"value": "sixty-nine"}
    return {"value": data[item_id]}

@app.get("/model-names")
def get_model_names():

    return {"response" : os.listdir(os.path.join("/code", "training"))}

@app.get("/get-number-of-output-folders/{model_name}")
def get_number_of_output_files(model_name : str):

    model_dir = os.path.join("/code", "training", model_name)
    if os.path.isdir(model_dir):
        model_subfolders = os.listdir(model_dir)
        output_count = 0
        for folder in model_subfolders:
            if folder.__contains__("output"):
                output_count += 1
        return {"response" : output_count}
    else:
        return {"response" : -1}

@app.get("/get-list-of-weights/{model_name}/{output}")
def get_list_of_weights(model_name : str, output : str):

    # Get path of directory weights should be in
    weights_dir = ""
    weights_dir = os.path.join("/code", "training", model_name, output, "weights")
    
    # Check if weights_dir exists. If it does, return a listing of all the .pt files in it
    if os.path.isdir(weights_dir):
        return {"response" : os.listdir(weights_dir)}
    
    # weights_dir does not exist. Check why it doesn't and return error message accordingly
    if os.path.isdir(os.path.join("code", "training", model_name)):
        return {"response" : f"Error: no such model with model name {model_name} exists"}
    if os.path.isdir(os.path.join("code", "training", model_name, output)):
        return {"response" : f"Error: output '{output}' does not exist"}
    return {"response" : f"Error: no weights directory in {output} folder"}

@app.get("/get-weights-file/{model_name}/{output}/{weights_name}")
def get_weights(model_name : str, output : str, weights_name : str):

    # Get path to desired weights
    weights_dir = os.path.join("/code", "training", model_name, output, "weights", weights_name)

    # Return weights file if it exists. Otherwise, return None
    if os.path.exists(weights_dir):
        with open(weights_dir, "rb") as file:
            file_content = file.read()
        return Response(content=file_content, media_type="application/octet-stream")
    return None



# POST METHODS

class TrainParameters(BaseModel):
    weights : str
    epochs : int
    batch_size : int
    img_size : int
    device : str
    sync_bn : bool
    workers : int
    name : str
@app.post("/train-model-yolov7")
async def train_model_yolov7(train_params : TrainParameters):

    data = os.path.join(os.getcwd(), "training", train_params.name, "data", "intrusion.yaml")
    cfg = os.path.join(os.getcwd(), "training", train_params.name, "cfg", "yolov7x.yaml")
    hyp = os.path.join(os.getcwd(), "training", train_params.name, 
                       "data", "hyp.scratch.custom.yaml")
    weights = os.path.join(os.getcwd(), "training", train_params.name, "pre-trained-weights",
                           train_params.weights)
    output = os.path.join(os.getcwd(), "training", train_params.name)

    cmd = ["python3", "yolov7/train.py"]
    cmd.extend(["--workers", f"{train_params.workers}"])
    cmd.extend(["--device", f"{train_params.device}"])
    if train_params.sync_bn:
        cmd.append("--sync-bn")
    cmd.extend(["--batch-size", f"{train_params.batch_size}"])
    cmd.extend(["--data", f"{data}"])
    cmd.extend(["--img-size", f"{train_params.img_size}"])
    cmd.extend(["--cfg", f"{cfg}"])
    cmd.extend(["--weights", f"{weights}"])
    cmd.extend(["--name", "output"])
    cmd.extend(["--project", f"{output}"])
    cmd.extend(["--hyp", f"{hyp}"])
    cmd.extend(["--epochs", f"{train_params.epochs}"])

    try:
        proc = await asyncio.create_subprocess_exec(*cmd)
        await proc.wait()
        return {"response": "Successfully trained model"}
    except Exception as e:
        return {"response": "Error while training model", "error": f"{e}"}

@app.post("/train-model-yolov9")
async def train_model_yolov9(train_params : TrainParameters):

    data = os.path.join(os.getcwd(), "training", train_params.name, "data", "intrusion.yaml")
    cfg = os.path.join(os.getcwd(), "training", train_params.name, "cfg", "yolov9-e.yaml")
    hyp = os.path.join(os.getcwd(), "training", train_params.name, 
                       "data", "hyp.scratch.custom.yaml")
    weights = os.path.join(os.getcwd(), "training", train_params.name, "pre-trained-weights",
                           train_params.weights)
    output = os.path.join(os.getcwd(), "training", train_params.name)

    cmd = ["python3", "yolov9/train_dual.py"]
    cmd.extend(["--workers", f"{train_params.workers}"])
    cmd.extend(["--device", f"{train_params.device}"])
    if train_params.sync_bn:
        cmd.append("--sync-bn")
    cmd.extend(["--batch-size", f"{train_params.batch_size}"])
    cmd.extend(["--data", f"{data}"])
    cmd.extend(["--img-size", f"{train_params.img_size}"])
    cmd.extend(["--cfg", f"{cfg}"])
    cmd.extend(["--weights", f"{weights}"])
    cmd.extend(["--name", "output"])
    cmd.extend(["--project", f"{output}"])
    cmd.extend(["--hyp", f"{hyp}"])
    cmd.extend(["--epochs", f"{train_params.epochs}"])

    try:
        proc = await asyncio.create_subprocess_exec(*cmd)
        await proc.wait()
        return {"response": "Successfully trained model"}
    except Exception as e:
        return {"response": "Error while training model", "error": f"{e}"}

class CreateDirectoryOptions(BaseModel):
    model_name : str
    copy_old_model_dataset : bool
    old_model_name : Union[str, None] = None
    copy_old_model_weights : bool
    old_model_weights_name : Union[str, None] = None
    model_type : str
@app.post("/create-new-model-directory", status_code=201)
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
        response += "\nAttempting to create new directories..."
        os.mkdir(new_cfg_path)
        os.mkdir(new_data_path)
        os.mkdir(new_dataset_path)
        os.mkdir(new_pretrained_weights_path)
        response += "\nCreated new directories"

        # Make alerts folder inside dataset folder
        new_alerts_path = os.path.join(new_dataset_path, "alerts")
        os.mkdir(new_alerts_path)

        # Get folder paths from rf_intrusion_yolov7x_v3 for folders to copy across
        v3_cfg_path = os.path.join(train_dir, "rf_intrusion_yolov7x_v3", "cfg")
        v3_pretrained_weights_path = os.path.join(
                train_dir,
                "rf_intrusion_yolov7x_v3", 
                "pre-trained-weights"
            )
        v3_data_path = os.path.join(train_dir, "rf_intrusion_yolov7x_v3", "data")
        
        # Copy data across from cfg and pre-trained-weights folders
        response += "\nAttempting to copy across cfg and pre-trained-weights folders..."
        try:
            if options.model_type == "yolov7":
                copy_directory_contents(v3_cfg_path, new_cfg_path)
                copy_directory_contents(
                        v3_pretrained_weights_path, 
                        new_pretrained_weights_path
                    )
            elif options.model_type == "yolov9":
                yolov9_std_weights_path = os.path.join(os.getcwd(), "yolov9", "weights")
                yolov9_std_cfg_path = os.path.join(os.getcwd(), "yolov9", "models", "detect")
                copy_directory_contents(yolov9_std_weights_path, new_pretrained_weights_path)
                copy_directory_contents(yolov9_std_cfg_path, new_cfg_path)
            else:
                response += "\nError: invalid model type selected"
                return {"response" : response}

        except Exception:
            response += "\nError: could not copy standard data across to new model"
            return {"response" : response}
        response += "\nCopied cfg and pre-trained-weights folders across"
        
        # Copy some of the files and info from data folders across
        response += "\nAttempting to create new intrusion.yaml file..."
        try:
            new_data_yaml_path = os.path.join(new_data_path, "intrusion.yaml")
            data  =  {
                "train" : os.path.join(new_data_path, 'train.txt'),
                "val" : os.path.join(new_data_path, 'val.txt'),  
                "test" : os.path.join(new_data_path, 'test.txt'),
                "nc" : 3,
                "names" : ['person', 'animal', 'vehicle']
            }
            with open(new_data_yaml_path, 'w') as yaml_file:
                yaml.dump(data, yaml_file, default_flow_style=False)
            response += "\nCreated new intrusion.yaml file"

        except Exception:
            response += "\nError: could not create intrusion.yaml file in data folder"
            return {"response" : response}
        
        # Copy across hyp.scratch.custom.yaml file
        try:
            shutil.copy2(os.path.join(v3_data_path, "hyp.scratch.custom.yaml"), 
                        os.path.join(new_data_path, "hyp.scratch.custom.yaml"))
        except Exception as e:
            response += f"\nError: could not copy across hyp.scratch.custom.yaml file\n{e}"
            return {"response" : response}
        
        # Try create train.txt, test.txt, and val.txt files
        response += "\nAttempting to create new train.txt and val.txt files..."
        try:
            train_txt_path = os.path.join(new_data_path, "train.txt")
            with open(train_txt_path, 'w') as train_txt:
                response += "\nCreated train.txt file"
            val_txt_path = os.path.join(new_data_path, "val.txt")
            with open(val_txt_path, 'w') as val_txt:
                response += "\nCreated new val.txt file"
            test_txt_path = os.path.join(new_data_path, "test.txt")
            with open(test_txt_path, 'w') as test_txt:
                response += "\nCreate new test.txt file"
        except Exception as e:
            response += f"Error: could not create train.txt and val.txt files.\n{e}"
            return {"response" : response}

    except FileNotFoundError:
        response += "\nError: could not create new directories"
        return {"response" : response}

    # Try copy the old dataset and data files across
    if options.copy_old_model_dataset:
        response += "\nAttempting to copy across dataset from previous model..."
        old_model_dir = os.path.join(train_dir, options.old_model_name)
        if os.path.exists(old_model_dir):
            # Try copy images from dataset across
            response += "\nAttempting to copy images across..."
            try:
                old_model_dataset_dir = os.path.join(
                                                old_model_dir, 
                                                "dataset", 
                                                "alerts"
                                            )
                copy_directory_contents(old_model_dataset_dir, new_dataset_path)
                response += "\nSuccessfully copied images across"
            except Exception:
                response += "\nError: could not copy dataset from old model across to new model"
                return {"response" : response}
            # Try copy data from train.txt and val.txt across
            response += "\nAttempting to copy across train.txt and val.txt files from old model..."
            try:
                old_model_data_dir = os.path.join(old_model_dir, "data")
                copy_text_file(old_model_data_dir, new_data_path, "train.txt")
                copy_text_file(old_model_data_dir, new_data_path, "val.txt")
            except Exception:
                response += "\nError: could not copy data from train.txt and val.txt across"
                return {"response" : response}
            response += "\nSuccessfully copied dataset across from old model"
        else:
            response += f"\nError: no model exists with the name '{options.old_model_name}'"
            return {"response" : response}

    # Try copy across the weights from a previous model
    if options.copy_old_model_weights:
        try:
            # Get old weights path
            old_weights_path = os.path.join(train_dir, 
                                            options.old_model_name, 
                                            "output", 
                                            "weights", 
                                            options.old_model_weights_name)

            # Check if old weights path leads to an actual file
            response += f"Attempting to copy old weights...\n\tPath = {old_weights_path}"
            if os.path.isfile(old_weights_path):
                copy_weights_path = os.path.join(
                        new_pretrained_weights_path, 
                        f"{options.old_model_name}.pt"
                    )
                shutil.copy2(old_weights_path, copy_weights_path)
            else:
                response += "\nError: no weights with specified name."
        except AttributeError:
            response += "\nError: no path given to old model's weights"
            return {"response" : response}
        except Exception as e:
            response += f"\nError: could not copy old weights across\n{e}"
            return {"response" : response}

    # Return response
    response = "All operations executed successfully"
    return {"response" : response}

@app.post("/upload-data/{model_name}")
def upload_data(model_name : str, files : List[UploadFile] = File(...)):

    response = ""
    try:
        data_path = os.path.join(os.getcwd(), "training", model_name, "data")
        # Iterate through posted files
        response += "Iterating through submitted files..."
        for file in files:
            # File is train.txt file
            if file.filename[-9:] == "train.txt":
                response += "\nFound train.txt file..."
                contents = file.file.read()
                filename = os.path.join(os.getcwd(), "training", model_name,
                                        "data", "train.txt")
                with open(filename, 'ab') as f:
                    response += f"\nAppending training data to {filename}"
                    f.write(contents)
            # File is the val.txt file
            elif file.filename[-7:] == "val.txt":
                response += "\nFound val.txt file..."
                contents = file.file.read()
                filename = os.path.join(os.getcwd(), "training", model_name,
                                        "data", "val.txt")
                with open(filename, 'ab') as f:
                    response += f"\nAppending validation data to {filename}"
                    f.write(contents)
            # File is the test.txt file
            elif file.filename[-8:] == "test.txt":
                response += "\nFound test.txt file..."
                contents = file.file.read()
                filename = os.path.join(os.getcwd(), "training", model_name,
                                        "data", "test.txt")
                with open(filename, 'ab') as f:
                    response += f"\nAppending testing data to {filename}"
                    f.write(contents)
            # File is an image or label file
            else:
                response += f"\nOpening {file.filename}..."
                filename = os.path.join(os.getcwd(), "training", model_name, 
                                   "dataset", "alerts", file.filename)
                with open(filename, 'wb') as f:
                    response += f"\nCopying {file.filename} to {filename}"
                    shutil.copyfileobj(file.file, f)
                    # shutil.copy2(f, dataset_path)
        response = "Successfully copied files across"
        return {"response" : response}
    except Exception as e:
        response += f"\n{e}"
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