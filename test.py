import cv2
import tensorflow as tf 
from model_aon import get_init_op
import os

flags = tf.app.flags
flags.DEFINE_string('exp_dir', 'exp_log', '')
flags.DEFINE_string('mode', 'single', '')
flags.DEFINE_string('image_path', '20.jpg', '')
flags.DEFINE_string('tags_file', '/share/zhui/svt1/test.tags', '')
FLAGS = flags.FLAGS


def load_image(image_path):
  image = cv2.imread(image_path)
  image = cv2.resize(image, (100, 100))
  image = image / 255.0
  return image

def test_single_picture():
  save_path = tf.train.latest_checkpoint(FLAGS.exp_dir)
  meta_file_path = save_path + '.meta'
  tf.reset_default_graph()
  saver = tf.train.import_meta_graph(meta_file_path)

  sess = tf.Session()
  sess.run(get_init_op())
  saver.restore(sess, save_path=save_path)  # restore sess

  graph = tf.get_default_graph()
  global_step = graph.get_tensor_by_name('global_step:0')
  image_placeholder = graph.get_tensor_by_name('input/Placeholder:0')
  output_eval_text_tensor = graph.get_tensor_by_name('attention_decoder/ReduceJoin_1:0')
  print('Restore graph from meta file {}'.format(meta_file_path))
  print('Restore model from {} successful, step {}'.format(save_path, sess.run(global_step)))
  if FLAGS.mode == 'single':
    pred_text = sess.run(output_eval_text_tensor, feed_dict={
      image_placeholder: load_image(FLAGS.image_path).reshape([1, 100, 100, 3])
    })
    print(pred_text)
  elif FLAGS.mode == 'tags':
    num_total = 0
    num_correct = 0
    with open(FLAGS.tags_file) as fo:
      for line in fo:
        try:
          image_path, gt = line.strip().split(' ')
          image = load_image(image_path)
        except Exception as e:
          print(e, image_path)
          continue
        pred_text = sess.run(output_eval_text_tensor, feed_dict={
          image_placeholder: image.reshape([1, 100, 100, 3])
        })
        print('{} ==> {}'.format(gt, pred_text))
        num_total += 1
        num_correct += (gt.lower() == pred_text[0].decode())
    print('Accu: {}/{}={}'.format(num_correct, num_total, num_correct/num_total))
  else:
    raise ValueError('Unsupported mode: {}'.format(FLAGS.mode))
  sess.close()


if __name__ == '__main__':
  test_single_picture()