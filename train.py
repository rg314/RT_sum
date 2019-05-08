import time
start = time.perf_counter()
import tensorflow as tf
import argparse
import pickle
import os
from model import Model
from utils import build_dict, build_dataset, batch_iter


# Uncomment next 2 lines to suppress error and Tensorflow info verbosity. Or change logging levels
# tf.logging.set_verbosity(tf.logging.FATAL)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def add_arguments(parser):
    parser.add_argument("--num_hidden", type=int, default=150, help="Network size.")
    parser.add_argument("--num_layers", type=int, default=2, help="Network depth.")
    parser.add_argument("--beam_width", type=int, default=10, help="Beam width for beam search decoder.")
    parser.add_argument("--glove", action="store_true", help="Use glove as initial word embedding.")
    parser.add_argument("--embedding_size", type=int, default=300, help="Word embedding size.")

    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--keep_prob", type=float, default=0.8, help="Dropout keep prob.")

    parser.add_argument("--toy", action="store_true", help="Use only 50K samples of data")

    parser.add_argument("--with_model", action="store_true", help="Continue from previously saved model")
# initalize parameters for model.


parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()
with open("args.pickle", "wb") as f:
    pickle.dump(args, f)

if not os.path.exists("output_data/saved_model"):
    os.mkdir("output_data/saved_model")
else:
    if args.with_model:
        old_model_checkpoint_path = open('output_data/saved_model/checkpoint', 'r')
        old_model_checkpoint_path = "".join(["output_data/saved_model/", old_model_checkpoint_path.read().splitlines()[0].split('"')[1]])
# check for old check points

print("Building dictionary...")
word_dict, reversed_dict, article_max_len, summary_max_len = build_dict("train", args.toy)
# build_dict for training data and test data
# 1) Re function used to further clean the text. This is why there is sometimes . . found in the input files as this is already done for data preperations
# 2) Words are tokenized using NLTK
# 3) Appends all tokenized words to a list. note that .mostcommon is called to find the most common tokens.
# 4) adds most commoned words to dict in the from of {token: word}
# 5) also forms reversed dict {word: token}
# 6) max length title = 15 and max length = article = 150. This is number of tokens


print("Loading training dataset...")
train_x, train_y = build_dataset("train", word_dict, article_max_len, summary_max_len, args.toy)
# 1) reads in txt train x and train y by reading text file
# 2) tokenizes words again for each article in txt file
# 3) looks at tokenized word and then loads dict to convert to number
# 4) cuts data based on max lengths that is token length!
# 5) dict with number. Note here is data isn't normalized


with tf.Session() as sess:
    model = Model(reversed_dict, article_max_len, summary_max_len, args)
    # makes a model object. and inputs the rev_dict plus placeholders for args
    sess.run(tf.global_variables_initializer())
    # initialized global varibles... Nothing more to say?
    saver = tf.train.Saver(tf.global_variables())
    #  sets up saver incase of a break
    if 'old_model_checkpoint_path' in globals():
        print("Continuing from previous trained model:", old_model_checkpoint_path, "...")
        saver.restore(sess, old_model_checkpoint_path)
    #  creates saver to log at check points
    batches = batch_iter(train_x, train_y, args.batch_size, args.num_epochs)
    # splits data into batches based on the sizes of the training data batch sizes and number of epochs
    # data hasn't be normaized yet.
    num_batches_per_epoch = (len(train_x) - 1) // args.batch_size + 1

    print("\nIteration starts.")
    print("Number of batches per epoch :", num_batches_per_epoch)
    for batch_x, batch_y in batches:
        batch_x_len = list(map(lambda x: len([y for y in x if y != 0]), batch_x))
        batch_decoder_input = list(map(lambda x: [word_dict["<s>"]] + list(x), batch_y))
        batch_decoder_len = list(map(lambda x: len([y for y in x if y != 0]), batch_decoder_input))
        batch_decoder_output = list(map(lambda x: list(x) + [word_dict["</s>"]], batch_y))

        batch_decoder_input = list(
            map(lambda d: d + (summary_max_len - len(d)) * [word_dict["<padding>"]], batch_decoder_input))
        batch_decoder_output = list(
            map(lambda d: d + (summary_max_len - len(d)) * [word_dict["<padding>"]], batch_decoder_output))

        train_feed_dict = {
            model.batch_size: len(batch_x),
            model.X: batch_x,
            model.X_len: batch_x_len,
            model.decoder_input: batch_decoder_input,
            model.decoder_len: batch_decoder_len,
            model.decoder_target: batch_decoder_output
        }

        # cells are created with a intial state all the encoding and decoding happens at this point.

        #         fw_cells = [self.cell(self.num_hidden) for _ in range(self.num_layers)]
        # bw_cells = [self.cell(self.num_hidden) for _ in range(self.num_layers)]
        # fw_cells = [rnn.DropoutWrapper(cell) for cell in fw_cells]
        # bw_cells = [rnn.DropoutWrapper(cell) for cell in bw_cells]

        # encoder_outputs, encoder_state_fw, encoder_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        #     fw_cells, bw_cells, self.encoder_emb_inp,
        #     sequence_length=self.X_len, time_major=True, dtype=tf.float32)
        # self.encoder_output = tf.concat(encoder_outputs, 2)
        # encoder_state_c = tf.concat((encoder_state_fw[0].c, encoder_state_bw[0].c), 1)
        # encoder_state_h = tf.concat((encoder_state_fw[0].h, encoder_state_bw[0].h), 1)
        # self.encoder_state = rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

        # This essentially creates the bidirectional component of the rnn. states of the fw and bw cells are concat together to give a state of the LSTM cell.

        # Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
        # and `h` is the output.

        # decoding uses BahdanauAttention model

        _, step, loss = sess.run([model.update, model.global_step, model.loss], feed_dict=train_feed_dict)

        # for each batch create a new dictionary and feed into network. Update and save weights after each update.

        if step % 1000 == 0:
            print("step {0}: loss = {1}".format(step, loss))
        if step % num_batches_per_epoch == 0:
            hours, rem = divmod(time.perf_counter() - start, 3600)
            minutes, seconds = divmod(rem, 60)
            saver.save(sess, "./output_data/saved_model/model.ckpt", global_step=step)
            print(" Epoch {0}: Model is saved.".format(step // num_batches_per_epoch),
                  "Elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds), "\n")
