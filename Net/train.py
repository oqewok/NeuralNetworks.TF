import tensorflow as tf
import data_reader
import conv_nn_faces_model as model


NUM_OF_EPOCHS = 10000
BATCH_COUNT = 10


def load_train_data():
    # здесь берем порциями данные
    names, labels = data_reader.read_labeled_image_list()
    data = data_reader.read_images_from_disk(names, labels)
    return data


def get_batched_data(data, BATCH_COUNT):
    img, masks = data[0], data[1]
    batched_img = [img[i:i + BATCH_COUNT] for i in range(0, len(img), BATCH_COUNT)]
    batched_masks = [masks[i:i + BATCH_COUNT] for i in range(0, len(masks), BATCH_COUNT)]
    return [batched_img, batched_masks]


def train():
    data = load_train_data()
    batched_data = get_batched_data(data, BATCH_COUNT)

    # model = get_model()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(NUM_OF_EPOCHS):
            batch = next_batch(batched_data, i % BATCH_COUNT)


def next_batch(batched_data, batch_index):
    i, l = batched_data[0], batched_data[1]
    images = i[batch_index, :]
    masks = l[batch_index, :]
    return [images, masks]