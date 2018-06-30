import tensorflow as tf
import label_map
import functools
import sync_attention_wrapper

def combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.
    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.
    Args:
        tensor: A tensor of any type.
    Returns:
        A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.shape(tensor)
    combined_shape = []
    for index, dim in enumerate(static_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_shape[index])
    return combined_shape


def _weight(shape, trainable=True, name='weights', initializer=None):
    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer()
    w = tf.get_variable(
        name=name, shape=shape, dtype=tf.float32, initializer=initializer, trainable=trainable
    )
    return w


def _bias(shape, trainable=True, name='biases', initializer=None):
    if initializer is None:
        initializer = tf.constant_initializer(0.0)
    b = tf.get_variable(
        name=name, shape=shape, dtype=tf.float32, initializer=initializer, trainable=trainable
    )
    return b


def _fc(layer_name, inputs, out_nodes):
    """
    Args:
        inputs: 4D, 3D or 2D tensor, if 4D tensor,
        out_nodes: number of output neutral units
    """
    shape = combined_static_and_dynamic_shape(inputs)
    if len(shape) == 4:
        size = shape[1] * shape[2] * shape[3]
    else:  # convert the last dimention to out_nodes size
        size = shape[-1]

    with tf.variable_scope(layer_name):
        w = _weight(shape=[size, out_nodes])
        b = _bias(shape=[out_nodes])
        flat_x = tf.reshape(inputs, [-1, size])
        x = tf.matmul(flat_x, w, name='matmul')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x)
        return x


def _conv(layer_name, inputs, out_channels, kernel_size=[3, 3], strides=[1, 1], paddings=[1, 1], trainable=True, reuse=None):
    """convolution layer with relu and batch normalization
    Args:
        layer_name: e.g. conv1, conv2
        x: input_tensor, [b, h, w, c]
        reuse: if reuse==tf.AUTO_REUSE: this convolution layer is parameter shared layer
    Returns:
        4D tensor
    """
    in_channels = combined_static_and_dynamic_shape(inputs)[-1]
    strides = [1, strides[0], strides[1], 1]
    p_h, p_w = paddings[0], paddings[1]
    paddings = [[0, 0], [p_h, p_h], [p_w, p_w], [0, 0]]

    with tf.variable_scope(layer_name, reuse=reuse):
        w = _weight(shape=[kernel_size[0], kernel_size[1], in_channels, out_channels], trainable=trainable)
        b = _bias(shape=[out_channels], trainable=trainable)
        x = tf.pad(inputs, paddings=paddings)
        x = tf.nn.conv2d(input=x, filter=w, strides=strides, padding='VALID', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        x = tf.layers.batch_normalization(inputs=x, axis=-1)  # In channel 
        return x


def _max_pool(layer_name, inputs, paddings, strides, ksize=[2, 2]):
    ksize = [1, ksize[0], ksize[1], 1]
    strides = [1, strides[0], strides[1], 1]
    p_h, p_w = paddings[0], paddings[1]
    paddings = [[0, 0], [p_h, p_h], [p_w, p_w], [0, 0]]

    with tf.variable_scope(layer_name):
        x = tf.pad(inputs, paddings=paddings)
        max_pool_ = tf.nn.max_pool(value=x, ksize=ksize, strides=strides, padding='VALID', name='max_pool')
        return max_pool_


def _bilstm(layer_name, inputs, hidden_units):
    with tf.variable_scope(layer_name):
        fw_lstm_cell = tf.contrib.rnn.LSTMCell(hidden_units)
        bw_lstm_cell = tf.contrib.rnn.LSTMCell(hidden_units)
        (output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
            fw_lstm_cell, bw_lstm_cell, inputs, dtype=tf.float32
        )
        output = tf.concat((output_fw, output_bw), 2)
        output_state_c = tf.concat((output_state_fw.c, output_state_bw.c), 1)
        output_state_h = tf.concat((output_state_fw.h, output_state_bw.h), 1)
        output_state = tf.contrib.rnn.LSTMStateTuple(output_state_fw, output_state_bw)
        return output, output_state


def base_cnn(x):
    """The basel convolutional neural network (BCNN) module for low-level visual representation
    Args:
        x, 4D tensor [b, w, h, c], w equal 100 and h equal 100 and channel equal 3
    """
    with tf.name_scope('BCNN') as scope:
        x = _conv(layer_name='conv_1', inputs=x, out_channels=64)
        x = _max_pool(layer_name='max_pool_1', inputs=x, strides=[2, 2], paddings=[0, 0])
        x = _conv(layer_name='conv_2', inputs=x, out_channels=128)
        x = _max_pool(layer_name='max_pool_2', inputs=x, strides=[2, 2], paddings=[1, 1])
        x = _conv(layer_name='conv_3', inputs=x, out_channels=256)
        x = _conv(layer_name='conv_4', inputs=x, out_channels=256)
        return x


def _arbitrary_orientation_network(inputs):
    """the arbitrary orientation network (AON) for capturing the horizontal, vertical and character placement features
    Args:
        feature_map, 4D tensor [b, w, h, c]
    """
    
    def get_character_placement_cluse(inputs):
        with tf.variable_scope('placement_cluse'):
            x = _conv(layer_name='conv_1', inputs=inputs, out_channels=512)
            x = _max_pool(layer_name='max_pool_1', inputs=x, strides=[2, 2], paddings=[1, 1])
            x = _conv(layer_name='conv_2', inputs=x, out_channels=512)
            x = _max_pool(layer_name='max_pool_2', inputs=x, strides=[2, 2], paddings=[1, 1])
            x = tf.reshape(x, shape=[-1, 64, 512])
            x = tf.transpose(x, perm=[0, 2, 1])
            x = _fc('fc_1', inputs=x, out_nodes=23)
            x = tf.reshape(x, shape=[-1, 512, 23])
            x = tf.transpose(x, perm=[0, 2, 1])
            x = _fc('fc_2', inputs=x, out_nodes=4)
            x = tf.reshape(x, shape=[-1, 23, 4])
            x = tf.nn.softmax(x, axis=2, name='softmax')
            return x

    def get_feature_sequence(inputs, reuse=None):
        with tf.variable_scope('shared_stack_conv', reuse=reuse):
            x = _conv(layer_name='conv_1', inputs=inputs, out_channels=512)
            x = _max_pool(layer_name='max_pool_1', inputs=x, strides=[2, 1], paddings=[1, 0])
            x = _conv(layer_name='conv_2', inputs=x, out_channels=512)
            x = _max_pool(layer_name='max_pool_2', inputs=x, strides=[2, 1], paddings=[0, 1])
            x = _conv(layer_name='conv_3', inputs=x, out_channels=512)
            x = _max_pool(layer_name='max_pool_3', inputs=x, strides=[2, 1], paddings=[1, 0])
            x = _conv(layer_name='conv_4', inputs=x, out_channels=512)
            x = _max_pool(layer_name='max_pool_4', inputs=x, strides=[2, 1], paddings=[0, 0])
            x = _conv(layer_name='conv_5', inputs=x, out_channels=512)
            x = _max_pool(layer_name='max_pool_5', inputs=x, strides=[2, 1], paddings=[0, 0])
            x = tf.squeeze(x, axis=1, name='squeeze')
            return x

    with tf.name_scope('AON_core') as scope:
        feature_horizontal = get_feature_sequence(inputs=inputs)
        feature_seq_1, _= _bilstm(layer_name='bilstm_1', inputs=feature_horizontal, hidden_units=256)
        feature_seq_1_reverse = tf.reverse(feature_seq_1, axis=[1])

        featute_vertical = get_feature_sequence(inputs=tf.image.rot90(inputs), reuse=True)
        feature_seq_2, _= _bilstm(layer_name='bilstm_2', inputs=featute_vertical, hidden_units=256)
        feature_seq_2_reverse = tf.reverse(feature_seq_2, axis=[1])

        character_placement_cluse = get_character_placement_cluse(inputs=inputs)
        
        res_dict = {
            'feature_seq_1': feature_seq_1,
            'feature_seq_1_reverse': feature_seq_1_reverse,
            'feature_seq_2': feature_seq_2,
            'feature_seq_2_reverse': feature_seq_2_reverse,
            'character_placement_cluse': character_placement_cluse,
        }
        return res_dict


def _filter_gate(aon_core_output_dict, single_seq=False):
    """the filter gate (FG) for combing four feature sequences with the character sequence.
    """
    feature_seq_1 = aon_core_output_dict['feature_seq_1']
    # DEBUG
    if single_seq:
        return feature_seq_1

    feature_seq_1_reverse = aon_core_output_dict['feature_seq_1_reverse']
    feature_seq_2 = aon_core_output_dict['feature_seq_2']
    feature_seq_2_reverse = aon_core_output_dict['feature_seq_2_reverse']
    character_placement_cluse = aon_core_output_dict['character_placement_cluse']

    with tf.name_scope('FG') as scope:
        A = feature_seq_1 * tf.tile(tf.reshape(character_placement_cluse[:, :, 0], [-1, 23, 1]), [1, 1, 512])
        B = feature_seq_1_reverse * tf.tile(tf.reshape(character_placement_cluse[:, :, 1], [-1, 23, 1]), [1, 1, 512])
        C = feature_seq_2 * tf.tile(tf.reshape(character_placement_cluse[:, :, 2], [-1, 23, 1]), [1, 1, 512])
        D = feature_seq_2_reverse * tf.tile(tf.reshape(character_placement_cluse[:, :, 3], [-1, 23, 1]), [1, 1, 512])
        res = A + B + C + D
        res = tf.tanh(res)
        return res



def _attention_based_decoder(encoder_outputs, groundtruth_text):
    batch_size = combined_static_and_dynamic_shape(encoder_outputs)[0]
    sync = True
    attention_wrapper_class = tf.contrib.seq2seq.AttentionWrapper if not sync else sync_attention_wrapper.SyncAttentionWrapper

    def decoder(helper, scope, batch_size, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            # attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=512, memory=encoder_outputs)
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=512, memory=encoder_outputs)
            cell = tf.contrib.rnn.GRUCell(num_units=512)
            attn_cell = attention_wrapper_class(
                cell, attention_mechanism, output_attention=False,
                attention_layer_size=256,
            )
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attn_cell, num_classes, reuse=reuse
            )
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=out_cell, helper=helper,
                initial_state=out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)  # batch_size
            )
            outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=100
            )
            return outputs[0]
    

    with tf.name_scope('attention_decoder'):
        GO_TOKEN = 0
        END_TOKEN = 1
        UNK_TOKEN = 2
        
        start_tokens = tf.fill([batch_size, 1], tf.constant(GO_TOKEN, tf.int64))
        end_tokens = tf.fill([batch_size, 1], tf.constant(END_TOKEN, tf.int64))

        label_map_obj = label_map.LabelMap()
        num_classes = label_map_obj.num_classes + 2
        embedding_fn = functools.partial(tf.one_hot, depth=num_classes)

        text_labels, text_lengths = label_map_obj.text_to_labels(groundtruth_text, pad_value=END_TOKEN, return_lengths=True)

        if not sync:
            train_input = tf.concat([start_tokens, start_tokens, text_labels], axis=1)
            train_target = tf.concat([start_tokens, text_labels, end_tokens], axis=1)
            train_input_lengths = text_lengths + 2
        else:
            train_input = tf.concat([start_tokens, text_labels], axis=1)
            train_target = tf.concat([text_labels, end_tokens], axis=1)
            train_input_lengths = text_lengths + 1

        max_num_step = tf.reduce_max(train_input_lengths)
        train_helper = tf.contrib.seq2seq.TrainingHelper(
            embedding_fn(train_input), tf.to_int32(train_input_lengths)
        )
        train_outputs = decoder(train_helper, 'decoder', batch_size)
        train_logits = train_outputs.rnn_output
        train_labels = train_outputs.sample_id
        weights=tf.cast(tf.sequence_mask(train_input_lengths, max_num_step), tf.float32)
        train_loss = tf.contrib.seq2seq.sequence_loss(
            logits=train_outputs.rnn_output, targets=train_target, weights=weights,
            name='train_loss'
        )
        train_probabilities = tf.reduce_max(
            tf.nn.softmax(train_logits, name='probabilities'),
            axis=-1,
        )
        train_output_dict =  {
            'loss': train_loss,
            'logits': train_logits,
            'labels': train_labels,
            'predict_text': label_map_obj.labels_to_text(train_labels),
            'probabilities': train_probabilities,
        }
        tf.summary.scalar(name='train_loss', tensor=train_loss)

        pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding_fn, 
            start_tokens=tf.fill([batch_size], GO_TOKEN),
            end_token=END_TOKEN
        )
        pred_outputs = decoder(pred_helper, 'decoder', batch_size, reuse=True)
        pred_logits = pred_outputs.rnn_output
        pred_labels = pred_outputs.sample_id
        eval_loss = tf.contrib.seq2seq.sequence_loss(
            logits=pred_outputs.rnn_output, targets=train_target, weights=weights,
            name='eval_loss'
        )
        pred_output_dict = {
            'logits': pred_logits,
            'labels': pred_labels,
            'predict_text': label_map_obj.labels_to_text(pred_labels),
        }
        return train_output_dict, pred_output_dict
            


def inference(images, groundtruth_text, single_seq=False):
    base_features = base_cnn(images)
    aon_core_output_dict = _arbitrary_orientation_network(base_features)
    encoded_sequence = _filter_gate(aon_core_output_dict, single_seq)
    train_output_dict, pred_output_dict = _attention_based_decoder(encoded_sequence, groundtruth_text)
    return train_output_dict, pred_output_dict


def get_train_op(loss, global_step):
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def get_init_op():
    init_op = tf.group(
        tf.local_variables_initializer(), 
        tf.global_variables_initializer(),
        tf.tables_initializer(),
    )
    return init_op


def test():
    images = tf.random_normal(shape=[2, 100, 100, 3], dtype=tf.float32)
    groundtruth_text = tf.constant(['this', 'company'], dtype=tf.string)
    output_tensor_dict = inference(images, groundtruth_text)
    # print(output_tensor_dict)


if __name__ == '__main__':
    test()
