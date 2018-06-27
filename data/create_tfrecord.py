import os 
import io
import tensorflow as tf 
import standard_fields as fields
import cv2 
import scipy
import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '/share/zhui/mnt/train.tfrecord', 'tfrecord filename')
flags.DEFINE_string('tags_file_path', '/share/zhui/mnt/ramdisk/max/imlist_filted.txt', 'tags file file')
FLAGS = flags.FLAGS


def main(unused_argv):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    count = 0

    with open(FLAGS.tags_file_path) as fo:
        for line in fo:
            image_path = line.strip()
            filename = '/'.join(line.strip().split('/')[-2:])
            groundtruth_text = line.split('_')[1]

            try:
                height, width, channel = cv2.imread(image_path).shape
                image_bin = open(image_path, 'rb').read()
            except Exception as e:
                print(e)
                continue
            
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    fields.TfExampleFields.image_encoded: dataset_util.bytes_feature(image_bin),
                    fields.TfExampleFields.height: dataset_util.int64_feature(height),
                    fields.TfExampleFields.width: dataset_util.int64_feature(width),
                    fields.TfExampleFields.filename: dataset_util.bytes_feature(filename.encode()),
                    fields.TfExampleFields.transcript: dataset_util.bytes_feature(groundtruth_text.encode())
                }
            ))

            writer.write(example.SerializeToString())
            count += 1
            if count % 1000 == 0:
                print(count)

    writer.close()
    print('{} example finished!'.format(count))
            


def make_tfrecord_from_tags(unused_argv):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    count = 0

    with open(FLAGS.tags_file_path) as fo:
        for line in fo:
            ts = line.split(' ')
            image_path = ts[0]
            groundtruth_text = ' '.join(ts[1:])

            height, width, channel = cv2.imread(image_path).shape
            assert channel == 3, '{} has {} channel'.format(image_path, channel)
            image_bin = open(image_path, 'rb').read()

            example = tf.train.Example(features=tf.train.Features(
                feature={
                    fields.TfExampleFields.image_encoded: dataset_util.bytes_feature(image_bin),
                    fields.TfExampleFields.height: dataset_util.int_feature(height),
                    fields.TfExampleFields.width: dataset_util.int_feature(width),
                    fields.TfExampleFields.transcript: dataset_util.bytes_feature(str(label).encode()),
                    fields.TfExampleFields.filename: dataset_util.bytes_feature(str(i).encode()),
                }
            ))

            writer.write(example.SerializeToString())
            count += 1

            if count % 1000 == 0:
                print(count)
        
    writer.close()
    print('{} example creater!'.format(count))


if __name__ == '__main__':
    tf.app.run()
