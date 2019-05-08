OLD_CHECKPOINT_FILE = "output_data/saved_model/model.ckpt-163700"
NEW_CHECKPOINT_FILE = "output_data/saved_model/model2.ckpt-163700"

import tensorflow as tf
vars_to_rename = {
    "stack_bidirectional_rnn/cell_10/bidirectional_rnn/bw/basic_lstm_cell/bias": "stack_bidirectional_rnn/cell_10/bidirectional_rnn/bw/basic_lstm_cell/bias",
}
new_checkpoint_vars = {}
reader = tf.train.NewCheckpointReader(OLD_CHECKPOINT_FILE)
for old_name in reader.get_variable_to_shape_map():
    if old_name in vars_to_rename:
        new_name = vars_to_rename[old_name]
    else:
        new_name = old_name
    new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))

init = tf.global_variables_initializer()
saver = tf.train.Saver(new_checkpoint_vars)
print(tf.global_variables())
with tf.Session() as sess:
    sess.run(init)
    saver.save(sess, NEW_CHECKPOINT_FILE)
