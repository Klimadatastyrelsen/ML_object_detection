# ML_object_detection

Maintained by the Danish Climate Data Agency for Bounding Box Detection on Oblique Images.

<img width="1124" height="1125" alt="image" src="https://github.com/user-attachments/assets/69aa5ca9-ac97-4a4b-98d5-b0820fc07f3c" />


## Installation

*   clone repo

    ```sh
    git clone https://github.com/SDFIdk/ML_object_detection
    ```
*   create conda environment
    ```sh
    cd ML_object_detection
    mamba env create -f environment.yml
    mamba activate  ML_object_detection
    ```  

## Example use
*   Use the included model for inference on one small and one large image from the included set of example images
    ```sh
    python src/ML_object_detection/infer_with_sahi.py --weights models/example_model.pt --folder_with_images data/example_images/ --result_folder output

    by placing the output .json files in the same folder as the image files you can inspect the result with labelme
    ```

## Download dataset from Hugging Face and train

The [KDS objects in oblique images](https://huggingface.co/datasets/rasmuspjohansson/KDS_objects_in_oblique_images) dataset is available on Hugging Face in YOLO format (images, labels, and `dataset.yaml`). To download it and train:

1. **Install the Hub client** (if not already installed):
   ```sh
   pip install huggingface_hub
   ```

2. **Download the dataset** (e.g. into a folder named `KDS_objects`):
   ```sh
   python -c "
   from huggingface_hub import snapshot_download
   snapshot_download(
       repo_id=\"rasmuspjohansson/KDS_objects_in_oblique_images\",
       repo_type=\"dataset\",
       local_dir=\"./KDS_objects\"
   )
   "
   ```
   Or clone the repo: `git clone https://huggingface.co/datasets/rasmuspjohansson/KDS_objects_in_oblique_images KDS_objects`

3. **Set the dataset path in `dataset.yaml`**  
   Open `KDS_objects/dataset.yaml` and set `path` to the directory that contains `images/` and `labels/` (the folder where you downloaded the data). For example, if you downloaded into `./KDS_objects`:
   ```yaml
   path: .    # use "." if you run train from inside KDS_objects, or use the full path to KDS_objects
   train: images/train
   val: images/val
   test:
   names:
     0: Velux
     1: Kvist
     2: Altan
   ```
   If you run training from the project root, set `path` to the absolute or relative path to `KDS_objects`, e.g. `path: ./KDS_objects`.

4. **Train** with the downloaded dataset:
   ```sh
   python src/ML_object_detection/train.py --data ./KDS_objects/dataset.yaml
   ```
   Useful flags (defaults match the windmill training setup where noted):

   | Flag | Default | Meaning |
   |------|---------|---------|
   | `--weights` | `yolov8n.pt` | Starting checkpoint |
   | `--epochs` | `100` | Training length |
   | `--imgsz` | `640` | Image size (match your chips) |
   | `--device` | `cuda:0` | GPU / CPU |
   | `--degrees` | `180` | Rotation augmentation ±degrees |
   | `--project` | `runs/detect` | Output root |
   | `--name` | `windmill_run` | Run folder; if it exists, Ultralytics uses `windmill_run2`, etc. |

   Example:
   ```sh
   python src/ML_object_detection/train.py \
     --data ./KDS_objects/dataset.yaml \
     --name my_experiment \
     --epochs 50
   ```

### Download labelme format and convert to YOLO for training

The same dataset is also provided in [labelme](https://github.com/wkentaro/labelme) format in the **labelme_format/** folder on the Hub (one `.json` and one `.tif` per image). To download that folder and convert it to YOLO so you can train:

1. **Download the dataset** (or clone the repo) so you have `labelme_format/` locally, e.g.:
   ```sh
   python -c "
   from huggingface_hub import snapshot_download
   snapshot_download(
       repo_id=\"rasmuspjohansson/KDS_objects_in_oblique_images\",
       repo_type=\"dataset\",
       local_dir=\"./KDS_objects\"
   )
   "
   ```
   Then your labelme files are in `KDS_objects/labelme_format/` (`.json` and `.tif` pairs).

2. **(Optional) Standardize JSON** if your labelme JSONs refer to different image paths or you need a consistent format:
   ```sh
   python src/ML_object_detection/standardize_json.py --json_dir ./KDS_objects/labelme_format
   ```

3. **Convert labelme to YOLO** with `labelme2yolo` (creates `images/`, `labels/`, and a config YAML):
   ```sh
   labelme2yolo --json_dir ./KDS_objects/labelme_format --val_size 0.15 --test_size 0.15
   ```
   By default the YOLO dataset is written under a subfolder of the directory that contains the JSON dir (see `labelme2yolo --help` for `--out_dir` if you want a specific path).

4. **Set the dataset path** in the generated `dataset.yaml` (e.g. `path: .` or the path to the folder that contains `images/` and `labels/`).

5. **Train** using the generated YOLO config:
   ```sh
   python src/ML_object_detection/train.py --data ./KDS_objects/YOLODataset/dataset.yaml
   ```
   (Adjust the path to `dataset.yaml` to match where `labelme2yolo` wrote it.)

## Create new dataset for training

* split the images to sizes suitable for yolo
  
    ```sh
    python split_with_gdal.py --image /path/to/large/images --output dataset/folder --x 640 --y 640 --overlap 40
    ```
*   Create a dataset with labelme
  
    Open folder containing the splitted images
    
    For each image you want to train on, draw rectangles from the upper left corner to the lower right corner.
    
    OBS. All objects of the categoriez you want to detect needs to be marked up. Partly labeled images will ruin the training.

*   (optional) set all "unkown"/"ignore" areas to black
  
    If you labeled areas with the text "ignore" (e.g areas for wich you are unsure about the correct classification) we have the option to mask all these areas and make them black.
    Calling mask_unknown_regions.py with -h flag will give more instructions on usage


* copy all data to a new location before doing the next steps

    note: draw rectangles from the upper left corner to the lower right corner

*   (optional) set all "unknown"/"ignore" areas to black:
    ```sh
    python src/ML_object_detection/mask_unknown_regions.py -h
    ```

*    make sure that all .json files use the same format (original .tif image)

    
    python standardize_json.py --json_dir /mnt/T/mnt/trainingdata/object_detection/from_Fdrev_ampol/all/
    
*   convert the labelme dataset to yolo format with 

    ```sh
    labelme2yolo --json_dir /path/to/labelme_json_dir/ --val_size 0.15 --test_size 0.15
    e.g labelme2yolo --json_dir /mnt/T/mnt/trainingdata/object_detection/from_Fdrev_ampol/split/
    ```
*   Train a object detection model on the dataset 

    ```sh
    python train.py --data /path/to/labelme_json_dir/config.yml
    e.g  python src/ML_object_detection/train.py --data /mnt/T/mnt/trainingdata/object_detection/object_detection_dataset/2025-06-16/labelme_images/YOLODataset/dataset.yaml
    ```    
*   Use the model for inference on large (unsplitted images) (e.g for creating sugestions for new labels) 

    ```sh
    python src/ML_object_detection/infer_with_sahi.py --weights models/example_model.pt --folder_with_images data/example_images/ --result_folder output

    ```    

