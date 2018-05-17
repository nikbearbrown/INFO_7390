from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image
from keras.models import load_model
from keras.applications import VGG16,InceptionV3

# This is to see which parts in an input image does a certain filter can recognize.

img_w, img_h = 139, 139

# Load saved model in Final_project_tf
model_path = './saved_model.h5'
model = load_model(model_path)
print('Model loaded')

# create a layer dictionary
input_img = model.input
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

# define a function to decode image
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# print a image from array
def show_feature(input_img_array):
    img = input_img_array[0]
    img = deprocess_image(img)
    return Image.fromarray(img, mode = 'RGB')

# get a sample image and decode
input_img_raw = image.load_img('./dataset_merged/training/soul/a1173693183_2.jpg', target_size=(139, 139))
input_img_array = image.img_to_array(input_img_raw)
input_img_array = np.expand_dims(input_img_array, axis = 0)

# define the name of layer to extract features from
layer_name = 'mixed3'
fig=plt.figure(figsize=(16, 8))
columns = 3
rows = 3

# get features from first 9 filters in the layer
index_start = 0
filter_num = 9

# loop through 9 filters
for filter_index in range(index_start,index_start+filter_num):
    print('Extracting features of filter %d in layer %s, %d/%d' %(filter_index,layer_name,filter_index-index_start+1,filter_num))
    input_copy = input_img_array.copy()
    # get the ouput of the layer
    layer_output = layer_dict[layer_name].output
    # define a loss funtion which is the mean value of the filter output
    loss = K.mean(layer_output[:, :, :, filter_index])
    # calculate gradients of layer.output/input_image
    grads = K.gradients(loss, input_img)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([input_img], [loss, grads])

    # Now we train our input to maximize the loss
    # In tuition, it's finding out which parts of the input image activate the filter most.
    step = 10  # 'learning rate'
    for i in range(30):
        # calculate loss and gradients for 30 times
        loss_value, grads_value = iterate([input_copy])
        # adjust input image to maximize the mean of layer output
        # this will make the recognized part become pure R/G/B pixels
        input_copy += grads_value * step

    # plot the 'trained' image
    fig.add_subplot(rows, columns, filter_index-index_start+1)
    plt.imshow(show_feature(input_copy))

plt.title('Filter %d~%d of layer %s' %(index_start,index_start+filter_num-1,layer_name))
fig.savefig('Inception_original_'+layer_name+' features.png')
print('Done')
plt.show()



"""
* License
This project is licensed under the MIT License - see the file [LICENSE.md](https://github.com/qiuminzhang/discogs_scrapy/blob/master/LICENSE) for details

* Citation
This project uses licensed open source python framework Scrapy - see the file [LICENSE.md](https://github.com/scrapy/scrapy/blob/master/LICENSE) for details.

For keras built-in Inception_v3 model, please refer to this https://keras.io/applications/#inceptionv3
"""



