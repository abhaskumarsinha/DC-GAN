# DC-GAN
A very simple and plain DC GAN to generate Image Flower Pictures out of dataset. 


## Dataset
Only a sample hundred data have been provided, which isn't enough to actually generate GAN Images. So, kindly full flower dataset and place it in `. /flowers` directory since: http://chaladze.com/l5/ (Linnaeus, 5 dataset Project).
We demonstrate MNIST Fashion Database for the same.

## Abstract
DC GAN or Deep Convolutional - GAN is a special case of GAN that uses multiple Convolutional Layers in deep nets in the form of Convolutional layers in Discriminator and Transpose Convolutional Layers in Generators (Basically they are reverse of Classical Convolutional Layers [1]). The basic advantage DC GANs provide over Classical GANs is tackling Mode Collapse during image training [2]. We use `tanh, sigmoid, relu, selu` activation functions provided from Keras in TensorFlow v2.0. 
**Please Note: The `Conv2DTranspose()` layer provided in Keras is different from Upsampling2D Layer, as the `Conv2DTranspose` layers' kernel weights are learnable during Training epochs. We will be using `Conv2DTranspose()` just for the same reason.** For training steps, kindly refer, 1D-GAN/Classic GAN repo for more details. We implement similar training steps here, except for the case, we train images by batch, `model.train_on_batch()` from Keras, which is an alternate and much simpler way to implement GANs without manually training weights with `gradient_tape()` as in Official Keras Documentation.

## Results

### Results on MNIST Fashion Dataset, 15 min run on Ryzen 5 CPU. Channel Size = 1 (Black and White) (`Total Samples = 60,000, each 28 x 28 x 1`)

https://user-images.githubusercontent.com/31654395/180400996-9b932a1b-9c62-42f2-bbfe-111d50a71e9c.mp4

### Results on Linnaeus 5 Flower Dataset, Training time = 1 hour 17 min on Google Colab GPU (`Total samples = 1,600, each 64 x 64 x 3`)

![GAN_1](https://user-images.githubusercontent.com/31654395/180401801-c61200ac-94ed-4470-b53c-aefaf79fad8a.png)
![GAN_2](https://user-images.githubusercontent.com/31654395/180401808-693a1b9b-098f-4808-993f-96ca255d3c8f.png)
![GAN_3](https://user-images.githubusercontent.com/31654395/180401822-745c888b-61c4-48cc-a5ab-53141f22d169.png)
![GAN_4](https://user-images.githubusercontent.com/31654395/180401830-b9cc1767-2e6f-4813-a2d6-87c3e5f01b91.png)
![GAN_5](https://user-images.githubusercontent.com/31654395/180401840-e108d4c9-8d7a-4568-9772-0d2f192fedfd.png)
![download (3)](https://user-images.githubusercontent.com/31654395/180401861-95f75fb8-da1d-4209-b164-4dd5222b58f6.png)
![download (4)](https://user-images.githubusercontent.com/31654395/180401874-2e8e16ab-353e-426a-bf18-e10ff7b954c5.png)
![download (5)](https://user-images.githubusercontent.com/31654395/180401882-896abb71-449e-4d07-908a-153f8acb066e.png)
![download (6)](https://user-images.githubusercontent.com/31654395/180401891-d6c43dfa-465b-4842-88f0-c1997a61da00.png)
![download (1)](https://user-images.githubusercontent.com/31654395/180401900-e53ba716-532d-471a-bb1d-e3d8ceee4456.png)
![download (2)](https://user-images.githubusercontent.com/31654395/180401918-d21a7082-95b9-4ce2-ba2a-1bd5ab5dcf6b.png)

As we can see from initial results, the GAN Model stared recognisising flowers in its initial stage itself. The shades, boundaries were apparently visible from start itself. The colors started converging to smooth and better gradually and it started converging to blurred flower visuals. 

For better results, train for 10-15 hours on Google Colab GPU for good convergence. 

##Biblography

1. Gao, Hongyang, et al. "Pixel transposed convolutional networks." IEEE transactions on pattern analysis and machine intelligence 42.5 (2019): 1218-1227.
2. Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
3. I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” in Advances in Neural Information Processing Systems, 2014, pp. 2672–2680
4. Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” Nature, vol. 521, no. 7553, pp. 436–444, 2015.
5. Thanh-Tung, Hoang, and Truyen Tran. "Catastrophic forgetting and mode collapse in gans." 2020 international joint conference on neural networks (ijcnn). IEEE, 2020.
6. Li, C.; Xu, K.; Zhu, J.; Zhang, B. Triple Generative Adversarial Nets. arXiv, 2016; arXiv:1703.02291.
