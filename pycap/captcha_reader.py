import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Dense, Activation
from keras.callbacks import ModelCheckpoint

path_to_images = "../data/captchas/"
saved_file_name = "../saved_models/weights.best.from_scratch"
avg_character_x_pixels = 22
x_window_length = avg_character_x_pixels * 1.5
character_start_position = 28
tolerance = 3
epochs = 20
all_characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                  'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

leaning_parameters = {
    "reload_weights": False, #This parameter if false, then re-runs the model over data, else it will load already available weights from saved_models folder
    "validation_function": "binary_crossentropy",
    "activation": "sigmoid",
    "single_char": True #This parameter determines if a single character is excised and fed to model or entire image is fed to model
}

activations = ['sigmoid', 'softmax']
validations = ['binary_crossentropy', 'categorical_crossentropy']
single_char = [False, True]


def get_save_file_name():
    return saved_file_name + "_" + leaning_parameters['validation_function'] + "_" + leaning_parameters[
        'activation'] + "_" + ("multi", "single")[leaning_parameters['single_char']]

# Returns the image array/vector with the path of the image
def get_image_vector_from_path(path):
    img = cv2.imread(path)
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.array(img_bw)


# Helper function to display image
def show_2d_image_vector(img_vector, title="Graph"):
    img_rgb = cv2.cvtColor(img_vector, cv2.COLOR_GRAY2BGR)
    plt.imshow(img_rgb)
    plt.title(title)
    plt.show()

def get_translated_images(input_image):
    shape = input_image.shape
    resultant_images = []
    for i in range(200-shape[1]):
        white_image = np.full(fill_value=255, shape=(50, 200))
        white_image[:, i:(i+shape[1])] = input_image
        resultant_images.append(white_image)
    return np.array(resultant_images)

# returns the image array and the text written in it
def get_image_data_with_labels(image_paths):
    image_array = []
    labels = []
    for image_name in image_paths:
        img_2d = get_image_vector_from_path(path_to_images + image_name)
        image_array.append(img_2d)

        label = image_name[:5]
        one_hot_output = np.zeros(36)
        for i in label:
            one_hot_output[all_characters.index(i)] = 1
        labels.append(one_hot_output)
    image_array = np.array(image_array)
    image_array = reshape_grayscale_images(image_array)
    labels = np.array(labels)
    return image_array, labels

# Returns image array for each character and corresponding character label
def get_single_char_img_with_labels(image_paths):
    image_array = []
    labels = []
    for image_name in image_paths:
        img_2d = get_image_vector_from_path(path_to_images + image_name)
        image_array.append(img_2d)
        label = image_name[:5]
        for i in label:
            one_hot_output = np.zeros(36)
            one_hot_output[all_characters.index(i)] = 1
            labels.append(one_hot_output)

    image_array = np.array(image_array)
    image_array = get_input_char_image_arr_from_image(image_array)
    labels = np.array(labels)
    return image_array, labels

# returns individual character image array
def get_input_char_image_arr_from_image(img_arr):
    image_array = []
    for j in range(len(img_arr)):
        img = img_arr[j]
        for i in range(5):
            cloned_img = np.copy(img)
            min_x, max_x = get_char_ranges(i)
            cloned_img[:, :int(min_x)] = 255
            cloned_img[:, int(max_x):] = 255
            image_array.append(cloned_img)
    image_array = np.array(image_array)
    return reshape_grayscale_images(image_array)


def get_char_ranges(num):
    min_x = character_start_position + num * avg_character_x_pixels
    max_x = min_x + avg_character_x_pixels + tolerance
    min_x = min_x - tolerance
    return min_x, max_x

# Converts grayscale images to convolution layer accepted array
def reshape_grayscale_images(gray_img_arr):
    shape = gray_img_arr.shape
    list_shape = list(shape)
    list_shape.append(1)
    return gray_img_arr.reshape(tuple(list_shape))

# Converts convolution accepted image to grayscale
def remove_dim_from_images(img_arr):
    return img_arr.reshape(img_arr.shape[:len(img_arr.shape) - 1])

# gets the character from an array which has 1 present for characters which are present
def get_characters_from_output_array(output):
    str = ""
    for i, val in enumerate(output):
        if val:
            str = str + all_characters[i]
    return str

# Train test data for entire image
def get_train_test_data():
    image_paths = os.listdir(path_to_images)
    inputs, outputs = get_image_data_with_labels(image_paths)
    return train_test_split(inputs, outputs, test_size=.20, random_state=42)

# Train test data for single character image
def get_train_test_data_for_single_char_image():
    image_paths = os.listdir(path_to_images)
    inputs, outputs = get_single_char_img_with_labels(image_paths)
    return train_test_split(inputs, outputs, test_size=.20, random_state=42)

def merge_all_images():
    images, labels = get_image_data_with_labels(os.listdir(path_to_images))
    images = remove_dim_from_images(images)
    final_image = np.zeros(shape=(50, 200))
    for image in images:
        final_image+=image
    final_image /= len(images)
    final_image = final_image.astype(np.uint8)
    show_2d_image_vector(final_image)

# Train test and validation data
def get_train_valid_test_data():
    input_train, input_test, output_train, output_test = \
        (get_train_test_data(), get_train_test_data_for_single_char_image())[leaning_parameters['single_char']]
    input_train, input_valid, output_train, output_valid = train_test_split(input_train, output_train, test_size=.20,
                                                                            random_state=42)
    return input_train, input_valid, input_test, output_train, output_valid, output_test


# Returns the learned model
def get_model():
    input_train, input_valid, input_test, output_train, output_valid, output_test = get_train_valid_test_data()

    # create CNN
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=4, strides=1, padding="valid", activation='relu',
                     input_shape=input_train[0].shape))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    Dropout(.2)
    model.add(Conv2D(filters=32, kernel_size=2, strides=1, padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    Dropout(.2)
    model.add(Conv2D(filters=64, kernel_size=2, strides=1, padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    Dropout(.2)
    model.add(Conv2D(filters=128, kernel_size=2, strides=1, padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(36))
    model.add(Activation(leaning_parameters['activation']))

    model.summary()

    print "Initializing compiling sequence"
    model.compile(optimizer='rmsprop', loss=leaning_parameters['validation_function'], metrics=['accuracy'])
    print("Compiled.")
    checkpointer = ModelCheckpoint(filepath=get_save_file_name(), verbose=1, save_best_only=True)

    if leaning_parameters['reload_weights']:
        model.load_weights(get_save_file_name())

    if not leaning_parameters['reload_weights']:
        model.fit(input_train, output_train,
                  validation_data=(input_valid, output_valid),
                  epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
    return model

# This function runs model over validation and test data.
def do_test_and_validate():
    model = get_model()

    input_train, input_valid, input_test, output_train, output_valid, output_test = get_train_valid_test_data()

    validation_batch = model.test_on_batch(input_valid, output_valid)
    testing_batch = model.test_on_batch(input_test, output_test)
    return validation_batch + testing_batch

# This function runs model for all configurations and prints the result. Basically loops over all the availalble parameters
def get_results_for_all_parameters():
    results = {}

    for i in activations:
        for j in validations:
            for k in single_char:
                leaning_parameters['activation'] = i
                leaning_parameters['validation_function'] = j
                leaning_parameters['single_char'] = k
                result = do_test_and_validate()
                results[i + "_" + j + "_" + str(k)] = result

    return results


# This function tries to read text by advancing window pixel by pixel and selecting a character whenever it crosses
# a threshold value, and restarting window
def get_image_text_with_cnn(image_file_name, model):
    min_x = 0
    max_x = 5
    step = 3
    threshold = .9
    name = ""
    image_array = get_image_vector_from_path(path_to_images + image_file_name)

    while max_x <= 200:
        cropped_image = np.copy(image_array)
        cropped_image[:, :min_x] = 255
        cropped_image[:, max_x:] = 255
        cropped_image = reshape_grayscale_images(cropped_image)
        prediction = model.predict(np.array([cropped_image]))[0]

        prediction.max()

        max_2 = prediction[np.argsort(prediction)[-2:]]
        if (max_2[1]-max_2[0])>0.30 and np.max(prediction) > threshold:
            min_x = max_x
            max_x += 5
            name += all_characters[np.argmax(prediction)]
        else:
            max_x+=step
        max_x+=step
        max_x += step
    return name

# This function tries to find character by assuming each character to be around avg-character size
def get_image_text_with_cnn_1(image_file_name, model):
    min_x = character_start_position
    max_x = character_start_position+avg_character_x_pixels
    name = ""
    image_array = get_image_vector_from_path(path_to_images + image_file_name)
    while max_x <=200:
        cropped_image = np.copy(image_array)
        cropped_image[:, :int(min_x-tolerance)] = 255
        cropped_image[:, int(max_x+tolerance):] = 255
        cropped_image = reshape_grayscale_images(cropped_image)
        prediction = model.predict(np.array([cropped_image]))[0]
        max_2 = prediction[np.argsort(prediction)[-2:]]
        if np.max(prediction) > .4 and (max_2[1]-max_2[0]) > .3:
            name+=all_characters[np.argmax(prediction)]
        min_x = max_x
        max_x += avg_character_x_pixels
    return name

# This function tries to read a character and then determines the correct size by gradually changing window size
def get_image_text_with_cnn_2(image_file_name, model):
    max_char_length = 25
    min_x = character_start_position
    max_x = character_start_position+max_char_length
    name = ""
    image_array = get_image_vector_from_path(path_to_images + image_file_name)
    while max_x <= 200 and len(name) !=5:
        cropped_image = np.copy(image_array)
        cropped_image[:, :int(min_x - tolerance)] = 255
        cropped_image[:, int(max_x + tolerance):] = 255
        cropped_image = reshape_grayscale_images(cropped_image)
        prediction = model.predict(np.array([cropped_image]))[0]
        name += all_characters[np.argmax(prediction)]
        min_x = get_char_boundary(image_array, model, min_x, np.argmax(prediction))
        max_x =min_x+avg_character_x_pixels
    return name

# Helper function which help determine character length
def get_char_boundary(image_array, model, min_x, char_index):
    max_x = min_x+18
    max_char_length = 25
    char_prob = 0
    while (max_x-min_x)<=max_char_length:
        cropped_image = np.copy(image_array)
        cropped_image[:, :int(min_x - tolerance)] = 255
        cropped_image[:, int(max_x + tolerance):] = 255
        cropped_image = reshape_grayscale_images(cropped_image)
        prediction = model.predict(np.array([cropped_image]))[0]
        if prediction[char_index]>char_prob:
            char_prob=prediction[char_index]
        else:
            if (max_x - min_x)==avg_character_x_pixels:
                return max_x
        max_x+=1
    return min_x+25

if __name__ == '__main__':
    images = os.listdir(path_to_images)
    model = get_model()

    count = 0
    for index, i in enumerate(images):
        image_name = i[:5]
        predicted_name = get_image_text_with_cnn_2(i, model)
        if image_name == predicted_name:
            count+=1
        else:
            print image_name +":" + predicted_name +":" + str(index)

    # Print count of captchas that are correct
    print count
