# DNN
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
![model](./assests/model.png)



## Training details (epoch < 200)
### accuracy
![loss_D_100](./assests/acc_graph.PNG)

### loss
![loss_G_100](./assests/loss_graph.PNG)


## Results
### test_accuracy
![test_acc](./assests/test_acc.PNG)

### test_loss
![test_loss](./assests/test_loss.PNG)



## Author
SangBeom-Hahn
