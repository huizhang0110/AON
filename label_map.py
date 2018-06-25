import tensorflow as tf

class LabelMap(object):

  def __init__(self,
               character_set=None,
               label_offset=2,
               ignore_case=True,
               unk_label=None):
    if character_set is None:
        character_set = list('abcdefghijklmnopqrstuvwxyz1234567890')
    if not isinstance(character_set, list):
      raise ValueError('character_set must be provided as a list')
    if len(frozenset(character_set)) != len(character_set):
      raise ValueError('Found duplicate characters in character_set')
    self._character_set = character_set
    self._label_offset = label_offset
    self._unk_label = unk_label or self._label_offset
    self._ignore_case = ignore_case

    print('Number of classes is {}'.format(self.num_classes))
    print('UNK label is {}'.format(self._unk_label))
    self._char_to_label_table, self._label_to_char_table = self._build_lookup_tables()

  @property
  def num_classes(self):
    return len(self._character_set)

  def _build_lookup_tables(self):
    chars = self._character_set
    labels = list(range(self._label_offset, self._label_offset + self.num_classes))
    char_to_label_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(
        chars, labels, key_dtype=tf.string, value_dtype=tf.int64),
      default_value=self._unk_label
    )
    label_to_char_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(
        labels, chars, key_dtype=tf.int64, value_dtype=tf.string),
      default_value=""
    )
    return char_to_label_table, label_to_char_table

  def text_to_labels(self,
                     text,
                     return_dense=True,
                     pad_value=-1,
                     return_lengths=False):
    """Convert text strings to label sequences.
    Args:
      text: ascii encoded string tensor with shape [batch_size]
      dense: whether to return dense labels
      pad_value: Value used to pad labels to the same length.
      return_lengths: if True, also return text lengths
    Returns:
      labels: sparse or dense tensor of labels
    """
    batch_size = tf.shape(text)[0]
    chars = tf.string_split(text, delimiter='')

    labels_sp = tf.SparseTensor(
      chars.indices,
      self._char_to_label_table.lookup(chars.values),
      chars.dense_shape
    )

    if return_dense:
      labels = tf.sparse_tensor_to_dense(labels_sp, default_value=pad_value)
    else:
      labels = labels_sp

    if return_lengths:
      text_lengths = tf.sparse_reduce_sum(
        tf.SparseTensor(
          chars.indices,
          tf.fill([tf.shape(chars.indices)[0]], 1),
          chars.dense_shape
        ),
        axis=1
      )
      text_lengths.set_shape([None])
      return labels, text_lengths
    else:
      return labels

  def labels_to_text(self, labels):
    """Convert labels to text strings.
    Args:
      labels: int32 tensor with shape [batch_size, max_label_length]
    Returns:
      text: string tensor with shape [batch_size]
    """
    if labels.dtype == tf.int32 or labels.dtype == tf.int64:
      labels = tf.cast(labels, tf.int64)
    else:
      raise ValueError('Wrong dtype of labels: {}'.format(labels.dtype))
    chars = self._label_to_char_table.lookup(labels)
    text = tf.reduce_join(chars, axis=1)
    return text



def test_label_map():
  label_map_obj = LabelMap()
  init_op = tf.group(
    tf.global_variables_initializer(), tf.local_variables_initializer(),
    tf.tables_initializer(),
  )
  test_string_tensor = tf.constant(['test', 'value', 'discombobulated', 'Chronographs', 'Chronographs'], dtype=tf.string)
  label_tensor, label_length_tensor = label_map_obj.text_to_labels(test_string_tensor, return_lengths=True)
  max_num_step = tf.reduce_max(label_length_tensor)
  label_length_mask_tensor = tf.cast(tf.sequence_mask(label_length_tensor, max_num_step), tf.float32)
  text_tensor = label_map_obj.labels_to_text(label_tensor)

  sess = tf.Session()
  sess.run(init_op)
  print('test_string', sess.run(test_string_tensor))
  print('label_tensor', sess.run(label_tensor))
  print('label_length', sess.run(label_length_tensor))
  print('label_length_mask', sess.run(label_length_mask_tensor))
  print('text_tensor', sess.run(text_tensor))
  sess.close()


if __name__ == '__main__':
  test_label_map()