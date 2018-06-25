import tensorflow as tf
import time
from model_aon import inference, get_init_op
from input_data import get_batch_data
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.python import debug as tfdbg


flags = tf.app.flags
flags.DEFINE_integer('batch_size', 32, 'define batch size')
flags.DEFINE_string('exp_dir', 'exp_log', 'Where you store checkpoint file')
flags.DEFINE_string('tfrecord_file_path', '/share/zhui/mnt/demo_5.tfrecord', 'tfrecord file path')
flags.DEFINE_string('tags_file', '/share/zhui/mnt/ramdisk/max/90kDICT32px/imlist.txt', '')
flags.DEFINE_string('data_dir', '/share/zhui/mnt/ramdisk/max/90kDICT32px/', '')
flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_integer('run_steps', 2, 'define batch size')
FLAGS = flags.FLAGS


def get_batch_data(batch_size=5):
  with open(FLAGS.tags_file) as fo:
    lines = fo.readlines()
  images = []
  groundtruth_texts = []
  while True:
    for line in lines:
      image_rel_path = line.strip().split('/', 1)[-1]
      image_abs_path = os.path.join(FLAGS.data_dir, image_rel_path)
      groundtruth_text = line.split('_')[1].lower()
      image = cv2.imread(image_abs_path)
      image = cv2.resize(image, (100, 100))
      image = image / 255.0
      images.append(image)
      groundtruth_texts.append(groundtruth_text)

      if len(images) == batch_size:
        yield {
          'images': images,
          'groundtruth_texts': groundtruth_texts
        }
        images = []
        groundtruth_texts = []


def test_get_batch_data():
  import time
  a = next(get_batch_data())
  images = a['images']
  groundtruth_texts = a['groundtruth_texts']
  for image, groundtruth_text in zip(images, groundtruth_texts):
    plt.imshow(image)
    print(groundtruth_text)
    plt.show()
    plt.clf()


def evaluation():
  save_path = tf.train.latest_checkpoint(FLAGS.exp_dir)
  meta_file_path = save_path + '.meta'
  tf.reset_default_graph()
  saver = tf.train.import_meta_graph(meta_file_path)
  
  sess = tf.Session()
  sess = tfdbg.LocalCLIDebugWrapperSession(sess) if FLAGS.debug else sess 
  sess.run(get_init_op())
  saver.restore(sess, save_path=save_path)  # restore sess
  
  graph = tf.get_default_graph()
  global_step = graph.get_tensor_by_name('global_step:0')
  image_placeholder = graph.get_tensor_by_name('input/Placeholder:0')
  groundtruth_placeholder = graph.get_tensor_by_name('input/Placeholder_1:0')
  output_eval_text_tensor = graph.get_tensor_by_name('attention_decoder/ReduceJoin_1:0')
  output_train_text_tensor = graph.get_tensor_by_name('attention_decoder/ReduceJoin:0')
  print('Restore graph from meta file {}'.format(meta_file_path))
  print('Restore model from {} successful, step {}'.format(save_path, sess.run(global_step)))

  batch_generator = get_batch_data(FLAGS.batch_size)
  for step in range(1, FLAGS.run_steps):
    batch_dict = next(batch_generator)
    images = batch_dict['images']
    groundtruth_texts = batch_dict['groundtruth_texts']
    print('generator {}'.format(len(images)))
    feed_eval = {
      image_placeholder: images
    }
    feed_train = {
      image_placeholder: images,
      groundtruth_placeholder: groundtruth_texts
    }

    eval_text = sess.run(output_eval_text_tensor, feed_eval)
    train_text = sess.run(output_train_text_tensor, feed_train)
    print('==STEP_{}=='.format(step))
    print('eval_text\n', eval_text)
    print('train_text\n', train_text)
    print('groundtruth_text\n', groundtruth_texts)
    print()
    print()
  sess.close()
  

def repeated_run_evaluation():
  '''
  每间隔一段时间,运行测试程序，并将测试结果写入到tensorboard中便于观察
  '''
  last_evaluation_model_path = None 
  eval_inteval_secs = 60 
  number_of_evaluation = 0
  while True:
    start_time = time.time()
    model_path = tf.train.latest_checkpoint(FLAGS.exp_dir)
    if not model_path:
      print('No model found in {}. Will try again in {} seconds.'.format(FLAGS.exp_dir, eval_inteval_secs))
    elif model_path == last_evaluation_model_path:
      print('Found already evaluated checkpoint. Will try again in {} secords'.format(eval_inteval_secs))
    else:
      last_evaluation_model_path = model_path
      pass 
    

if __name__ == '__main__':
  evaluation()
  # test_get_batch_data()
