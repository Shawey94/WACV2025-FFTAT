# [Feature Fusion Transferability Aware Transformer for Unsupervised Domain Adaptation](https://arxiv.org/pdf/2411.07794), WACV 2025

### updates (11/10/2024)
<!--  Add the environment requirements to reproduce the results.  --> 

<p align="left"> 
<img width="800" src="https://github.com/Shawey94/FFTAT/blob/main/image.png">
</p>

### Environment (Python 3.8.12)
```
# Install Anaconda (https://docs.anaconda.com/anaconda/install/linux/)
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh

# Install required packages
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -c pytorch
pip install tqdm==4.50.2
pip install tensorboard==2.8.0
# apex 0.1
conda install -c conda-forge nvidia-apex
pip install scipy==1.5.2
pip install timm==0.6.13
pip install ml-collections==0.1.0
pip install scikit-learn==0.23.2
```

### Pretrained ViT
Download the following models and put them in `checkpoint/`
- ViT-B_16 [(ImageNet-21K)](https://storage.cloud.google.com/vit_models/imagenet21k/ViT-B_16.npz?_ga=2.49067683.-40935391.1637977007)
- ViT-B_16 [(ImageNet)](https://console.cloud.google.com/storage/browser/_details/vit_models/sam/ViT-B_16.npz;tab=live_object)


<!-- 
TVT with ViT-B_16 (ImageNet-21K) performs a little bit better than TVT with ViT-B_16 (ImageNet):
<p align="left"> 
<img width="500" src="https://github.com/uta-smile/TVT/blob/main/ImageNet_vs_ImageNet21K.png">
</p>
 --> 

### Datasets:

- Download images from [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw), [VisDA-2017](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification), [DomainNet](https://ai.bu.edu/M3SDA/#dataset) and put them under `data/`. For example, images of `Office-31` should be located at `data/office/domain_adaptation_images/`, please refer to the format in 'data/office' and 'data/office-home'. The file size of DomainNet and Visda2017 is too big for GitHub, so you can only find 'office' and 'office-home' under 'data/'.

### Training:

All commands can be found in `script.txt`. An example:
```
python3 main.py --train_batch_size 16 --dataset office --name wa \
--source_list data/office/webcam_list.txt --target_list data/office/amazon_list.txt \
--test_list data/office/amazon_list.txt --num_classes 31 --model_type ViT-B_16 \
--pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256 \
--beta 0.1 --gamma 0.2 --use_im --theta 0.1
```

<!-- 
### Attention Map Visualization:
```
python3 visualize.py --dataset office --name wa --num_classes 31 --image_path att_visual.txt --img_size 256
```
The code will automatically use the best model in `wa` to visualize the attention maps of images in `att_visual.txt`. `att_visual.txt` contains image paths you want to visualize, for example:
```
/data/office/domain_adaptation_images/dslr/images/calculator/frame_0001.jpg 5
/data/office/domain_adaptation_images/dslr/images/calculator/frame_0002.jpg 5
/data/office/domain_adaptation_images/dslr/images/calculator/frame_0003.jpg 5
/data/office/domain_adaptation_images/dslr/images/calculator/frame_0004.jpg 5
/data/office/domain_adaptation_images/dslr/images/calculator/frame_0005.jpg 5
```
 --> 

### Citation:
```
@article{yu2024FFTAT,
  title={Feature Fusion Transferability Aware Transformer for Unsupervised Domain Adaptation},
  author={Yu, Xiaowei and Huang, Zhe and Zhang, Zao},
  journal={arXiv preprint arXiv:2411.07794},
  year={2024}
}

@article{yu2023RCCT,
  title={Robust core-periphery constrained transformer for domain adaptation},
  author={Yu, Xiaowei and Zhu, Dajiang and Liu, Tianming},
  journal={arXiv preprint arXiv:2308.13515},
  year={2023}
}
```
Our code is largely borrowed from [TVT](https://github.com/uta-smile/TVT) and [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
