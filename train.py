import tensorflow as tf 
from model_aon import inference, get_train_op, get_init_op
from input_data import get_batch_data
import os
import numpy as np

flags = tf.app.flags
flags.DEFINE_string('exp_dir', 'exp_log', 'experiment model save directory')
flags.DEFINE_integer('batch_size', 32, 'define train batch size')
flags.DEFINE_integer('max_steps', 2000000, 'step nums for training')
flags.DEFINE_boolean('restore', True, 'restore model parameter from checkpoint file')
flags.DEFINE_string('tfrecord_file_path', '/share/zhui/mnt/train.tfrecord', 'tfrecord file path')
FLAGS = flags.FLAGS


def main(unused_argv):
    if FLAGS.exp_dir:
        checkpoint_dir = os.path.join(FLAGS.exp_dir, 'model.ckpt')
        train_log_write_dir = os.path.join(FLAGS.exp_dir, 'log/train')

    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.name_scope('input'):
        image_placeholder = tf.placeholder(shape=[None, 100, 100, 3], dtype=tf.float32)
        groundtruth_text_placeholder = tf.placeholder(shape=[None,], dtype=tf.string)
        tf.summary.image('input_image', image_placeholder, FLAGS.batch_size)
    print('image_placeholder', image_placeholder)
    print('groundtruth_placeholder', groundtruth_text_placeholder)
    
    output_tensor_dict, eval_output_tensor_dict = inference(image_placeholder, groundtruth_text_placeholder)
    loss_tensor = output_tensor_dict['loss']
    output_labels_tensor = output_tensor_dict['labels']
    output_predict_text_tensor = output_tensor_dict['predict_text']
    print('output_predict_text_tensor', output_predict_text_tensor)
    probabilities_tensor = output_tensor_dict['probabilities']

    output_eval_text_tensor = eval_output_tensor_dict['predict_text']  # For EVAL
    print('output_eval_text_tensor', output_eval_text_tensor)

    train_op = get_train_op(loss_tensor, global_step)
    batch_tensor_dict = get_batch_data(FLAGS.tfrecord_file_path, mode='train', batch_size=FLAGS.batch_size)

    decoder_inputs_tensor = tf.get_default_graph().get_tensor_by_name("attention_decoder/concat:0")
    decoder_targets_tensor = tf.get_default_graph().get_tensor_by_name("attention_decoder/concat_1:0")

    sess = tf.Session()
    train_writer = tf.summary.FileWriter(train_log_write_dir, sess.graph)
    summary_merge_tensor = tf.summary.merge_all()
    sess.run(get_init_op())

    total_loss = 0.0
    begin_step = 0
    saver = tf.train.Saver()

    if os.path.exists(os.path.join(FLAGS.exp_dir, 'checkpoint')) and FLAGS.restore:
        save_path = tf.train.latest_checkpoint(FLAGS.exp_dir)
        saver.restore(sess, save_path=save_path)
        begin_step = sess.run(global_step)
        print('Restore model from {} successful, continue training from step {}'.format(save_path, begin_step))
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    try:
        for step in range(begin_step, FLAGS.max_steps):
            if coord.should_stop():
                break
            batch_dict = sess.run(batch_tensor_dict)
            images = batch_dict['images']
            groundtruth_text = np.char.lower(batch_dict['groundtruth_text'].astype('str'))
            feed_dict = {image_placeholder: images, groundtruth_text_placeholder: groundtruth_text}
            _, loss = sess.run([train_op, loss_tensor], feed_dict=feed_dict)
            total_loss += loss 

            if step % 100 == 0:
                summary, output_labels, output_predict_text, decoder_inputs, decoder_targets= sess.run(
                    [summary_merge_tensor, output_labels_tensor, output_predict_text_tensor, decoder_inputs_tensor, decoder_targets_tensor],
                    feed_dict=feed_dict
                )
                probabilities = sess.run(probabilities_tensor, feed_dict)
                eval_text = sess.run(output_eval_text_tensor, feed_dict={image_placeholder: images})
                train_writer.add_summary(summary, step)

                print('Step {}, loss {}'.format(step, total_loss / 100))
                print('out_labels\n', output_labels[:5])
                print('predict_text\n', output_predict_text[:5])
                print('probabilities\n', probabilities[:5])

                print('groundtruth_text\n', groundtruth_text[:5])
                print('decoder_inputs\n', decoder_inputs[:5])
                print('decoder_targets\n', decoder_targets[:5])
                print('eval_text\n', eval_text[:5])
                sample_image = images[:1]
                print('Use a sample: ', sess.run(output_eval_text_tensor, feed_dict={image_placeholder: sample_image}))
                print()
                print()
                total_loss = 0.0
            
            if step % 1000 == 0:
                saver.save(sess, save_path=checkpoint_dir, global_step=global_step)
                print('Write checkpoint')
                
    except tf.errors.OutOfRangeError():
        print('All finished')
    finally:
        coord.request_stop()
        coord.join(threads)
    sess.close()


if __name__ == '__main__':
    tf.app.run()

