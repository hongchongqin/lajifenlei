#对终端输入的数据进行归类
import tensorflow as tf
import numpy as np
import os, sys
import data_input_helper as data_helpers
import jieba

# Parameters

# Data Parameters
#数据参数
#定义一个用于接收 string 类型数值的变量
#参数：变量名称、默认值、用法描述
tf.flags.DEFINE_string("w2v_file", "./data/word2vec.bin", "w2v_file path")

# Eval Parameters
#评估参数
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/checkpoints/", "Checkpoint directory from training run")

# Misc Parameters
#其他参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


#用于帮助我们添加命令行的可选参数。也就是说可以不用反复修改源代码中的参数，
#而是利用该函数可以实现在命令行中选择需要设定或者修改的参数来运行程序。

FLAGS = tf.flags.FLAGS#FLAGS保存命令行参数的数据
#FLAGS._parse_flags()
FLAGS.flag_values_dict()#将其解析成字典存储到FLAGS.flag_values_dict中

class RefuseClassification():

    def __init__(self):
    
        self.w2v_wr = data_helpers.w2v_wrapper(FLAGS.w2v_file)#加载词向量,通过data_input_helper.py处理数据
        self.init_model()
        self.refuse_classification_map = {0: '可回收垃圾', 1: '有害垃圾', 2: '湿垃圾', 3: '干垃圾'}
        
        
    def deal_data(self, text, max_document_length = 10):
        
        words = jieba.cut(text)#jieba分词
        x_text = [' '.join(words)]#把分出来的词用空格隔开
        x = data_helpers.get_text_idx(x_text, self.w2v_wr.model.vocab_hash, max_document_length)
		#使用data_input_helper.py里的get_text_idx函数
        return x


    def init_model(self):
        #寻找最近保存的FLAGS
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        graph = tf.Graph()#计算图，主要用于构建网络，本身不进行任何实际的计算。
        with graph.as_default():
			#创建会话
            session_conf = tf.ConfigProto(
                              allow_soft_placement=FLAGS.allow_soft_placement, #记录设备指派情况
                              log_device_placement=FLAGS.log_device_placement) #选择运行设备
            self.sess = tf.Session(config=session_conf)
            self.sess.as_default()
            # Load the saved meta graph and restore variables
			#加载保存的元图和还原变量
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(self.sess, checkpoint_file)

            # Get the placeholders from the graph by name
			#从图表中按名称获取占位符
            self.input_x = graph.get_operation_by_name("input_x").outputs[0]
          
            self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
			#我们要计算的张量
            self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]
                
    
    def predict(self, text):
    
        x_test = self.deal_data(text, 5)
		
        predictions = self.sess.run(self.predictions, {self.input_x: x_test, self.dropout_keep_prob: 1.0})
        refuse_text = self.refuse_classification_map[predictions[0]]
        return refuse_text


if __name__ == "__main__":
    if len(sys.argv) == 2:
        test = RefuseClassification()
        res = test.predict(sys.argv[1])
        print('classify:', res)
