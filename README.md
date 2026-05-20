


# GeoPro: Geometric-based Prototypical Feature Space Expansion

This repo contains the implementation of GeoPro, an improvement from MEMO, a class-incremental learning method.

To run the repo on CIFAR100 or ImageNet100 dataset, please refer to the following scripts.

<div align="center">
  <img src="resources/memo.png" width="90%">


</div>


## Prerequisites
- [torch](https://github.com/pytorch/pytorch)
- [torchvision](https://github.com/pytorch/vision)
- [tqdm](https://github.com/tqdm/tqdm)
- [numpy](https://github.com/numpy/numpy)


## Training scripts
- Train CIFAR100
```
python main_memo.py -model memo -init 10 -incre 10 -ms 3312 -net memo_resnet32 -p fair -d 3 --train_base -d 0 1 2 3
```
To run the code on Imagenet100, you MUST download the ImageNet100 dataset and put the link to the dataset in the `utils\data` file.
link to download ImageNet100: https://www.kaggle.com/datasets/ambityga/imagenet100
1. sau khi tải dataset về từ kaggle, cần unzip:
```
unzip imagenet100.zip -d /root/data/imagenet100/
```
rồi gộp các thư mục train.X thành 1 thư mục train duy nhất:
```
for dir in /root/data/imagenet100/train.X*/; do
    cp -r "$dir"*/ /root/data/imagenet100/train/
done
```
Lưu ý sửa lại đường dẫn trên cho đúng với khi chạy thực tế.

2. sửa tên mục val.X thành val

3. truy cập vào ICLR23-MEMO, vào utils/data, sửa lại link đến dataset imagenet vừa tạo, comment dòng <code>assert 0, "You should specify ..."</code> lại và sửa đường dẫn <code>train_dir, test_dir</code> cho phù hợp

- Train Imagenet100
```
python main_memo.py --dataset imagenet100 --model_name memo -convnet_type memo_resnet18 --init_cls 10 --increment 10 --device 0 --batch_size 64
```
