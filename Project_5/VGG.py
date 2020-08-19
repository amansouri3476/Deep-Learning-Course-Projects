import tensorflow as tf
from matplotlib import pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model

# noinspection SpellCheckingInspection
model = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
                                          pooling=None, classes=1000)

# ############################################# Problem 3 - Part 2 #############################################

print(model.summary())

# ############################################# Problem 3 - Part 3 #############################################

layer_dict = dict([(layer.name, layer) for layer in model.layers])

# 64 Channel Convolution Layers

for layer_name in ['block1_conv1', 'block1_conv2']:

    layer_w = layer_dict[layer_name].get_weights()[0][:, :, 0, :]

    plt.clf()
    plt.figure(figsize=(20, 20))
    for i in range(64):
        plt.subplot(8, 8, i+1)
        plt.imshow(layer_w[:, :, i])

    plt.savefig(layer_name + '.png')

# 128 Channel Convolution Layers

for layer_name in ['block2_conv1', 'block2_conv2']:

    layer_w = layer_dict[layer_name].get_weights()[0][:, :, 0, :]

    plt.clf()
    plt.figure(figsize=(20, 20))
    for i in range(128):
        plt.subplot(16, 8, i + 1)
        plt.imshow(layer_w[:, :, i])

    plt.savefig(layer_name + '.png')

for layer_name in ['block3_conv1', 'block3_conv2', 'block3_conv3']:

    layer_w = layer_dict[layer_name].get_weights()[0][:, :, 0, :]

    plt.clf()
    plt.figure(figsize=(20, 20))
    for i in range(256):
        plt.subplot(16, 16, i + 1)
        plt.imshow(layer_w[:, :, i])

    plt.savefig(layer_name + '.png')

for layer_name in ['block4_conv1', 'block4_conv2', 'block4_conv3']:

    layer_w = layer_dict[layer_name].get_weights()[0][:, :, 0, :]

    plt.clf()
    plt.figure(figsize=(20, 20))
    for i in range(512):
        plt.subplot(32, 16, i + 1)
        plt.imshow(layer_w[:, :, i])

    plt.savefig(layer_name + '.png')

for layer_name in ['block5_conv1', 'block5_conv2', 'block5_conv3']:

    layer_w = layer_dict[layer_name].get_weights()[0][:, :, 0, :]

    plt.clf()
    plt.figure(figsize=(20, 20))
    for i in range(512):
        plt.subplot(32, 16, i + 1)
        plt.imshow(layer_w[:, :, i])

    plt.savefig(layer_name + '.png')

# ############################################ Problem 3 - Part 4 #############################################

model = VGG16()
plt.figure(figsize=(40, 40))

for pic_name in [
    'brown_bear',
    'cat_dog',
    'dd_tree',
    'dog_beagle',
    'scenery',
    'space_shuttle'
]:
    for layer_name in [
        'block1_pool',
        'block5_pool'
    ]:

        image = load_img('Images/' + pic_name + '.png', target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        y = model.predict(image)
        label = decode_predictions(y)
        label = label[0][0]

        print('%s (%.2f%%)' % (label[1], label[2] * 100))

        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        layer_output = layer_dict[layer_name].output

        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=layer_output)
        intermediate_output = intermediate_layer_model.predict(image)

        if layer_name == 'block1_pool':
            plt.clf()
            for i in range(64):
                plt.subplot(8, 8, i + 1)
                plt.imshow(intermediate_output[0, :, :][:, :, i])
            plt.savefig(pic_name + '_output:' + layer_name + '.png')

        if layer_name == 'block5_pool':
            plt.clf()
            for i in range(512):
                plt.subplot(32, 16, i + 1)
                plt.imshow(intermediate_output[0, :, :][:, :, i])
            plt.savefig(pic_name + '_output:' + layer_name + '.png')
