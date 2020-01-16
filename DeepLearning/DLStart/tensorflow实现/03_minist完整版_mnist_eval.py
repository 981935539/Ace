import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

# 加载的时间间隔。
EVAL_INTERVAL_SECS = 10

# 定义神经网络相关参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH="MNIST_model/"
MODEL_NAME="mnist_model"


# 通过tf.get_variable 函数来获取变量。在训练神经网络时会创建这些变量；在测试时会通过保存的模型加载这些变量的取值。
# 而且更加方便的是，因为可以在变量加载时将滑动平均变量重命名，所以可以直接通过同样的名字在训练时使用变量自身，
# 而在测试时使用变量的滑动平均值。
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights


# 定义变量的前向传播过程
def inference(input_tensor, regularizer):
    # 这里使用tf.get variable 或 tf.Variable 没有本质区别 ，因为在训练或是测试中
    # 没有在同一个程序中多次调用这个函数。如果在同一个程序中多次调用,在第一次调用之后需要将reuse参数设置为True.
    with tf.variable_scope('layer1'):

        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        # 直接通过调用装好的函数来计算前向传播的结果。因为测试时不关注正则化损失的值，
        # 所以这里用于计算正则化损失的函数被设置为None.
        y = inference(x, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 这里通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数
        # 获取平均值了。这样就可以完全共用inference中定义的前向传播过程。
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # 每隔EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中正确率的变化。
        while True:
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state 函数会通过 checkpoint 文件自动找到目录中最新模型的文件名。
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的轮数。
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets("../dataset/MNIST_data/", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()


