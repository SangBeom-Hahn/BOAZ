# Vanilla_GAN
Tensorflow implementation of DNN

## Requirements
* tensorflow 2.x
* python 3.x

## Core code
```python
model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(N_HIDDEN,
   		input_shape=(RESHAPED,),
   		name='dense_layer', activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))

model.add(keras.layers.Dense(N_HIDDEN,
   		name='dense_layer_2', activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))

model.add(keras.layers.Dense(NB_CLASSES,
   		name='dense_layer_3', activation='sigmoid'))
```


## Model
![model](./assests/graph.png)

```python
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
loss_G = tf.reduce_mean(tf.log(D_gene))
```
## Training details (epoch < 100)
### loss_D
![loss_D_100](./assests/loss_D_100.JPG)

### loss_G
![loss_G_100](./assests/loss_G_100.JPG)

## Training details (epoch < 1000)
### loss_D
![loss_D](./assests/loss_D.JPG)

### loss_G
![loss_G](./assests/loss_G.JPG)

*Even though each of loss_D and loss_G are maximized, loss_D and loss_G are related to each other, so the two loss values will not always tend to increase together.*

*If loss_D increases, loss_G should decrease, and if loss_G increases, loss_D should decrease.*

*Because it is an adversarial relationship.*

## Results
### epoch=0
![epoch_0](./samples/000.png)

### epoch=100
![epoch_100](./samples/100.png)

### epoch=300
![epoch_300](./samples/300.png)

### epoch=600
![epoch_600](./samples/600.png)

### epoch=900
![epoch_900](./samples/900.png)

## Author
Junho Kim
