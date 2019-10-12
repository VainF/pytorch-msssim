# Train an autoencoder with SSIM & MS-SSIM

## Prepare dataset
1. Download CLIC datase from https://www.compression.cc/challenge/ and unzip them into datasets/data

2. Download validation set (Kodak)
    ```bash 
    bash download_kodak.sh
    ```

3. The structure of the directory:

    ```
    datasets/  
        data/
            CLIC/
                train/
                val/
                mobile_train.zip
                mobile_valid.zip
                professional_train.zip
                professional_valid.zip
            kodak/
                kodim01.png
                kodim02.png
                ...
    ...
    ```

## Train
```bash
python train.py --loss_type ssim
python train.py --loss_type ms_ssim
```
