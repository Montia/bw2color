how to use pretrained vgg19 model in this project?

First, keras provide some pretrained models, include vgg19

- [keras doc for pretrained model usage](https://keras.io/applications/#available-models)

Then, how to use keras models in tensorflow?

- Blog [Use Keras Pretrained Models With Tensorflow](Use Keras Pretrained Models With Tensorflow)
and its [code](https://github.com/zachmoshe/zachmoshe.com/blob/master/content/use-keras-models-with-tf/using-keras-models-in-tf.ipynb)
is great reference for use keras pretrained model in tensorflow

and how to modify code to get dense 2048 fc1 layer without relu?

- [source code](https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py)
 in keras_applications/vgg19.py
 
- [style2paint-keras](https://github.com/harrywang-1523/style2paint-keras) repo use [this](https://github.com/fchollet/deep-learning-models/blob/master/vgg19.py)
as basic code and modify to get dense 2048 fc1 layer without relu
[code](https://github.com/harrywang-1523/style2paint-keras/blob/master/keras_vgg19.py)
 
