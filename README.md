# Light-DehazeNet
Light-DehazeNet: A Novel Lightweight CNN Architecture for Single Image Dehazing

![](readme_images/Picture1.png)

# Introduction
This repo contains the implementation of our proposed LD-Net, the prerequisite libraries, and link of the dataset we have used in our experiments. For testing purpose we also provide a trained LD-Net weights and test hazy images. 

----
# Dependencies
This code requires the following libraries:
- torch==1.9.1
- torchvision==0.10.1
- numpy==1.19.2
- PIL==8.2.0
- matplotlib==3.3.4

run ```pip install -r requirements.txt``` to install all the dependencies. 

# Training Dataset
For training we have used the Synthetic Hazy Dataset generated by [Boyi Li](https://sites.google.com/site/boyilics/website-builder/project-page) and his team for single image dehazing task.

- The synthetic hazy images [training set](https://drive.google.com/file/d/17ZWJOpH1AsYQhoqpWR6PK61HrUhArdAK/view)
- The original images [training set](https://drive.google.com/file/d/1Sz5ZFFZXo3sY85R3v7yJa6W6riDGur46/view)

# Citation
Please cite our paper, if you want to reproduce the results using this code.
```
@article{ullah2021light,
  title={Light-DehazeNet: A Novel Lightweight CNN Architecture for Single Image Dehazing},
  author={Ullah, Hayat and Muhammad, Khan and Irfan, Muhammad and Anwar, Saeed and Sajjad, Muhammad and Imran, Ali Shariq and De Albuquerque, Victor Hugo C},
  journal={IEEE Transactions on Image Processing},
  year={2021},
  publisher={IEEE}
}
```

```
Ullah, H., Muhammad, K., Irfan, M., Anwar, S., Sajjad, M., Imran, A. S., & De Albuquerque, V. H. C., 
"Light-DehazeNet: A Novel Lightweight CNN Architecture for Single Image Dehazing", 
IEEE Transactions on Image Processing, 2021.
```
