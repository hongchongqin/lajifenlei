'''
CNN卷积神经网络（TextCNN 文本分类的卷积神经网络）
步骤如下：
1.嵌入层（输出层）：输入一个定长的文本序列，经过一系列处理，最终输入层输入的是文本序列中各个词汇对应的分布式表示。
2.卷积层+池化层（Max-pool(最大值池化)：即每个卷积核的最大值集中输出）
3.使用DropOut防止过拟合
4.输出模型的预测结果即分数
'''
#coding=utf-8
import tensorflow as tf
import numpy as np

class TextCNN(object):#卷积神经网络
    
    #A CNN for text classification.
    #Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
	#用于文本分类的CNN
	#使用一个嵌入层，然后是卷积层、最大池层和softmax层。
	#CNN的每次卷积操作需要输入几个字向量，这就相当于在计算词的特征，将文本从字特征转到更高级的词特征。
	#cnn网络处理文本的理解，可以把卷积层看作n-gram的一种处理。每一句话可以当做一个图像问题。
	#卷积就是对每个词的上下文提取特征。
	#输入->基础特征提取->多层复杂特征提取->权重学习->预测结果
    
    def __init__(
      self,
	  w2v_model, 
	  sequence_length,
	  num_classes, #类别数
	  vocab_size, #字典大小
      embedding_size,  #词向量维度
	  filter_sizes, #卷积核大小
	  num_filters,  #卷积核数目
	  l2_reg_lambda=0.0
	  ):

        # Placeholders for input, output and dropout
		#输入、输出和退出的占位符
		#placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。
        self.input_x = tf.compat.v1.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
		#跟踪l2正则化损失（可选）
        l2_loss = tf.constant(0.0)

        # Embedding layer
		#嵌入层
		#CNN-rand（随机初始化）
		
		#指定词向量的维度embedding_size后，文本分类模型对不同单词的向量作随机初始化,
		#后续有监督学习过程中，通过BP的方向更新输入层的各个词汇对应的词向量。
		
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if w2v_model is None:
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="word_embeddings")#W矩阵的初始化（W是我们在训练中学习的嵌入矩阵）
            else:
                self.W = tf.compat.v1.get_variable("word_embeddings",
                    initializer=w2v_model.vectors.astype(np.float32))

            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)#创建实际的嵌入操作
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
		#为每个卷积核大小创建一个卷积+maxpool层
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
				#卷积层
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1),dtype=tf.float32, name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]),dtype=tf.float32, name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,#W是我们的滤波器矩阵
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
				#应用非线性
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
				#输出上的最大池
				
				#在Text-CNN模型的池化层中使用了Max-pool（最大值池化），
				#即减少模型的参数，又保证了在不定长的卷基层的输出上获得一个定长的全连接层的输入。
				
                pooled = tf.nn.max_pool2d(
                    h,#h是将非线性应用于卷积输出的结果
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)#获得一个最大值的卷积核

        # Combine all the pooled features
		#合并所有池功能
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
		#防止过拟合
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
           W = tf.get_variable(
                        "W",
                        shape=[num_filters_total, num_classes],
                        initializer=tf.contrib.layers.xavier_initializer())
           b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
           l2_loss += tf.nn.l2_loss(b)         
           self.scores = tf.compat.v1.nn.xw_plus_b(self.h_drop, W, b, name="scores")
           self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
		#最终（未标准化）分数和预测
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
		#准确度
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


  
