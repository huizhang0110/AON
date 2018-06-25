import model_aon
import tensorflow as tf
import input_data
import numpy as np


def classfier(images):
  x = model_aon.base_cnn(images)
  x = model_aon._fc('classfier_fc', x, 10)
  preds = tf.argmax(x, axis=1)

  return {
    'preds': preds,
    'logits': x
  }

def main():
  batch_size = 32
  images_placeholder = tf.placeholder(shape=[None, 100, 100, 3], dtype=tf.float32)
  labels_placeholder = tf.placeholder(shape=[None], dtype=tf.int64)
  output_tensor_dict = classfier(images_placeholder)
  logits_tensor, preds_tensor = output_tensor_dict['logits'], output_tensor_dict['preds']
  loss_tensor = tf.reduce_mean(
    tf.losses.sparse_softmax_cross_entropy(
      logits=logits_tensor,
      labels=labels_placeholder,
    )
  )
  train_op = tf.train.AdadeltaOptimizer().minimize(loss_tensor)
  batch_tensor_dict = input_data.get_batch_data()


  sess = tf.Session()

  init_op = tf.group(
    tf.global_variables_initializer(), tf.local_variables_initializer()
  )
  sess.run(init_op)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord, sess=sess)

  try:
    for step in range(1, 100000):
      if coord.should_stop():
        break
        
      batch_dict = sess.run(batch_tensor_dict)
      feed_dict = {
        images_placeholder: batch_dict['images'],
        labels_placeholder: batch_dict['groundtruth_text'].astype(np.int64)
      }
      _, loss = sess.run([train_op, loss_tensor], feed_dict=feed_dict)

      if step % 100 == 0:
        print('step {} loss {}'.format(step, loss))
        preds = sess.run(preds_tensor, feed_dict=feed_dict)
        labels = batch_dict['groundtruth_text']
        print('preds\n', preds[:10])
        print('labels\n', labels[:10])
  except tf.errors.OutOfRangeError():
    print('All finished')
  finally:
    coord.request_stop()
    coord.join(threads)
  
  sess.close()

  


if __name__ == '__main__':
  main()