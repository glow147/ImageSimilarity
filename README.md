## Environment

```
docker pull pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
```

## Summary
This model is for getting image similarity (Cosine Similarity)

* Model Information  
I used MobileNet for feature extract from image.  
Pretrained Weight was not loaded to observe whether the correct Loss Function was used and the model's learning process.  

* Loss Function  
Loss Functions are consist of Cosine Similarity and Triplet Loss(L2)

* Load Dataset  
Dataset got from Huggingface.

## Method
### About args
(--configs, default="configs/default.yaml") : Load model config from yaml file.  
(--dataset, default="imagenet-1k") : Setting Dataset.  
(--loss, choices=['cos', 'L2'], default='cos') : Setting Loss function.  
(--output_dir, default="results") : Save path for Model ckpt.  
(--save_image, default=False) : If you want to save Sample Image, declare this statement.  

### Train Code
```
python main.py --loss {'L2', 'cos'} --save_image
```
### Resume Train Code
```
python main.py --resume --weight {model_ckpt_path}
```

### Validation/Test Code
```
python main.py --test --weight {model_ckpt_path}

example: python main.py --test --weight ./best.ckpt
```
