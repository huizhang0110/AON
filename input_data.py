import tensorflow as tf
import cv2
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def read_tfrecord_use_pythonAPI(filename):
    example_iter = tf.python_io.tf_record_iterator(path=filename)  # yield string generator
    count = 0
    for r in example_iter:  # r is a byte flow
        example = tf.train.Example()
        example.ParseFromString(r)  # make a proto message
        label = example.features.feature['image/transcript'].bytes_list.value[0].decode()
        image_bin = example.features.feature['image/encoded'].bytes_list.value[0]
        filename = example.features.feature['image/filename'].bytes_list.value[0].decode()
        buf = io.BytesIO()
        buf.write(image_bin)
        image = Image.open(buf)
        image.show()
        buf.seek(0)
        count += 1

        if count > 10:
            break
    print('read finished')


def test_python_api():
    tfrecord_file = '/share/zhui/mnt/demo_20.tfrecord'
    read_tfrecord_use_pythonAPI(tfrecord_file)


def read_tfrecord_use_queue_runner(filename, batch_size=32):
    filequeue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, example_tensor = reader.read(filequeue)  # return ther next key/value pair produced by a reader
    # 注意read是一个死循环，不要直接使用 sess.run(key or value) key和value都是一个string tensor
    # %%1: Use tf.parse_single_example to parse example tensor
    example_features = tf.parse_single_example(
        example_tensor,
        features={
            'image/transcript': tf.FixedLenFeature([], dtype=tf.string),
            'image/height': tf.FixedLenFeature([], dtype=tf.int64),
            'image/width': tf.FixedLenFeature([], dtype=tf.int64),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string),
            'image/filename': tf.FixedLenFeature([], dtype=tf.string)
        }
    )
    height = tf.cast(example_features['image/height'], tf.int32)
    width = tf.cast(example_features['image/width'], tf.int32)

    image = tf.image.decode_jpeg(example_features['image/encoded'], channels=3)
    image = tf.reshape(image, [height, width, 3])
    image = tf.image.resize_images(image, [100, 100]) / 128.0 - 1  # normalize to [-1, 1)

    groundtruth_text = tf.cast(example_features['image/transcript'], tf.string)    
    filename = tf.cast(example_features['image/filename'], tf.string)
    # %%2: make a batch
    min_after_dequeue = 2000
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, groundtruth_text_batch, filename_batch= tf.train.shuffle_batch(
        [image, groundtruth_text, filename],
        batch_size=batch_size, 
        min_after_dequeue=min_after_dequeue,
        capacity=capacity,
        num_threads=8
    )

    batch_tensor_dict = {
        'filenames': filename_batch,
        'images': image_batch,
        'groundtruth_text': groundtruth_text_batch,
    }

    return batch_tensor_dict


def test_queue_runner():
    tfrecord_file = '/share/zhui/mnt/demo_20.tfrecord'
    batch_tensor_dict = read_tfrecord_use_queue_runner(tfrecord_file, batch_size=10)
    sess = tf.Session()
    init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    max_time_steps = 1
    try:
        for step in range(max_time_steps):
            if coord.should_stop():
                break
            batch_dict = sess.run(batch_tensor_dict)
            images = batch_dict['images']
            labels = batch_dict['groundtruth_text']
            filenames = batch_dict['filenames']
            for i, (image, label) in enumerate(zip(images, labels)):
                # image = np.squeeze(image, axis=2)
                plt.subplot(5, 2, i + 1)
                plt.imshow(image)
                print(label)
            plt.show()

    except tf.errors.OutOfRangeError():
        print('Done training')
    finally:
        coord.request_stop()  # send stop message
        coord.join(threads)  # wait for all 

    sess.close()
    exit()


def get_batch_data(tfrecord_path, batch_size=32, mode='train'):
    if mode=='train':
        return read_tfrecord_use_queue_runner(tfrecord_path, batch_size=batch_size)
    else:
        raise ValueError('Unsupported mode: {}'.format(mode))


if __name__ == '__main__':
    # test_queue_runner()
    test_python_api()
    
