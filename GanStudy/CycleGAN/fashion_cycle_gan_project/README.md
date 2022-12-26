# CycleGAN
sketch2fashion of CycleGAN

## Requirements
* tensorflow 2.x
* python 3.x

## Core code
```python

# 생성자 선언
class CycleGAN(CycleGAN):
    def build_generator(self):
        """U-Net 생성자"""
        # 이미지 입력
        d0 = Input(shape=self.img_shape)

        # 다운샘플링
        d1 = self.conv2d(d0, self.gf)
        d2 = self.conv2d(d1, self.gf * 2)
        d3 = self.conv2d(d2, self.gf * 4)
        d4 = self.conv2d(d3, self.gf * 8)

        # 업샘플링
        u1 = self.deconv2d(d4, d3, self.gf * 4)
        u2 = self.deconv2d(u1, d2, self.gf * 2)
        u3 = self.deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4,
                            strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)

# 판별자 선언
class CycleGAN(CycleGAN):
    def build_discriminator(self):
      img = Input(shape=self.img_shape)

      d1 = self.conv2d(img, self.df, normalization=False) #필터 개수를 두배씩 늘린다.
      d2 = self.conv2d(d1, self.df * 2)
      d3 = self.conv2d(d2, self.df * 4)
      d4 = self.conv2d(d3, self.df * 8)

      validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

      return Model(img, validity)
    
```


## Generator
![model](./assests/generator.png)


## Results
### epoch=5
![test_acc](./assests/epoch5.png)

### epoch=10
![test_acc](./assests/epoch10.png)

### epoch=20
![test_acc](./assests/epoch20.png)

### epoch=50
![test_acc](./assests/epoch50.png)

## Author
SangBeom-Hahn
