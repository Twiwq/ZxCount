# ZxCount:Mask-Guided Hierarchical Memory Alignment for Cross-Domain Crowd Counting

This is an official repository for our work, "ZxCount:Mask-Guided Hierarchical Memory Alignment for Cross-Domain Crowd Counting"
## Requirements
* Python 3.10.12
* PyTorch 2.0.1
* Torchvision 0.15.2
* Others specified in [requirements.txt](requirements.txt)

## Data Preparation
1. Download ShanghaiTech and UCF-QNRF datasets from official sites and unzip them.
2. Run the following commands to preprocess the datasets:
    ```
    python utils/preprocess_data.py --origin-dir [path_to_ShanghaiTech]/part_A --data-dir data/sta
    python utils/preprocess_data.py --origin-dir [path_to_ShanghaiTech]/part_B --data-dir data/stb
    python utils/preprocess_data.py --origin-dir [path_to_UCF-QNRF] --data-dir data/qnrf
    ```
3. Run the following commands to generate GT density maps:
    ```
    python dmap_gen.py --path data/sta
    python dmap_gen.py --path data/stb
    python dmap_gen.py --path data/qnrf
    ```

## Training
Run the following command:
```
python main.py --task train --config configs/sta_train.yml
```
You may edit the `.yml` config file as you like.

## Testing
Run the following commands after you specify the path to the model weight in the config file:
```
python main.py --task test --config configs/sta_test_stb.yml
python main.py --task test --config configs/sta_test_qnrf.yml
```

## Inference
Run the following command:
```
python inference.py --img_path [path_to_img_file_or_directory] --model_path [path_to_model_weight] --save_path output.txt --vis_dir vis
```

## Pretrained Weights
We provide pretrained weights in the table below:
| Source | Performance                                   | Weights                                                                                                                                          |
| ------ | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| B      | A: 88.7MAE, 158.1MSE<br>Q: 157.7MAE, 264.8MSE | [Google Drive](https://drive.google.com/file/d/1ygkz3wc8jjs58BNQWSgzX8s3zfn_j1Pi/view?usp=sharing) |

