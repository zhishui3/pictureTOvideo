import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn

class Data:
    def __init__(self,data_size,num_batch,batch_size,time_step):
        self.data_size = data_size #数据集大小
        self.batch_size = batch_size
        self.num_batch = num_batch
        self.time_step = time_step#rnn的时间步
        self.data_without_rel = [] #保存随机生成的数据，数据间没有联系
        self.data_with_rel = [] # 保存有时序关系的数据
		#np.random.choice”函数生成的由0和1组成的长串数据
    def generate_data(self):
	    self.data_without_rel = np.array(np.random.choice(2,size=(self.data_size,)))
        for i in range(self.data_size):
            if self.data_without_rel[i-1] == 1 and self.data_without_rel[i-2] == 1:
#之前连续出现两个1，当前数据设为0
			    self.data_with_rel.append(0)
				continue
			elif self.data_without_rel[i-1] == 0 and self.data_without_rel[i-2] == 0:
			    #之前连续出现两个0，当前数据设为1
				self.data_with_rel.append(1)
				continue
				#np.random.rand()产生的随机数范围：[0,1]
			else:
			    if np.random.rand() >=0.5:
				    self.data_with_rel.append(1)
			    else:
				    self.data_with_rel.append(0)

					
        return self.data_without_rel,self.data_with_rel
		
	#有了数据我们接下来要用RNN去学习这些数据，看看它能不能学习到我们产生这些数据时使用的策略，即数据间的联系。评判RNN是否学习到规律以及学习的效果如何的依据，是我们在第三章里介绍过的交叉熵损失函数。根据我们生成数据的规则，如果RNN没有学习到规则，那么它预测正确的概率就是0.5，否则它预测正确的概率为：0.5*0.5+0.5*1=0.75（在“data_without_rel”中，连续出现的两个数字的组合为：00、01、10和11。00和11出现的总概率占0.5，在这种情况下，如果RNN学习到了规律，那么一定能预测出下一个数字，00对应1，11对应0。而如果出现的是01或10的话，RNN预测正确的概率就只有0.5，所以综合起来就是0.75）。
	
	#根据交叉熵损失函数，在没有学习到规律的时候，其交叉熵损失为：
      #  loss = - (0.5 * np.log(0.5) + 0.5 * np.log(0.5)) = 0.6931471805599453
     #在学习到规律的时候，其交叉熵损失为：

 #Loss = -0.5*(0.5 * np.log(0.5) + np.log(0.5))
 #=-0.25 * (1 * np.log(1) ) - 0.25 * (1 *np.log(1))=0.34657359027997264   

    def generate_epochs(self):
	    self.generate_data()
		data_x = np.zeros([self.num_batch,self.batch_size],dtype=np.int32)
		data_y = np.zeros([self.num_batch,self.batch_size],dtype=np.int32)
		
		#将数据划分成num_batch
		for in range(self.num_batch):
		    data_x[i] = self.data_without_rel[self.batch_size*i:self.batch_size*(i-1)]
			data_y[i] = self.data_with_rel[self.batch_size*i:self.batch_size*(i+1)]
		#将每个batch的数据按time_step进行切分
		epoch_size = self.batch_size // self.time_step
		
		#返回最终的数据
		for i in range(epoch_size):
		    x = data_x[:,self.time_step*i:self.time_step*(i+1)]
			y = data_y[:,self.time_step*i:self.time_step*(i+1)]
			yield(x,y)
			
			
class Model:
    def __init__(self,data_size,batch_size,time_step,state_size)；
	    self.data_size = data_size
		self.batch_size =batch_size
		self.num_batch = self.data_size // self.batch_size
		self.time_step = time_step
		self.state_size = state_size
		
	#输入数据的占位符
	self.x = tf.placeholder(tf.int32,[self.num_batch,self.time_step],name='input_placeholder')
	self.y = tf.placeholder(tf.int32,[self.num_batch,self.time_step],name='labels_placeholder')
	
	#记忆单元的占位符
	self.init_state = tf.zeros([self.num_batch,self.state_size])
	#将输入数据进行one-hot编码
	self.rnn_inputs = tf.one_hot(self.x,2)
	
	#隐藏层的权重矩阵和偏置项
	self.W = tf.get_variable('W',[self.state_size,2])
	self.b = tf.get_variable('b',[2],initializer=tf.constant_initializer(0.0))
	
	#RNN隐藏层的输出
	self.rnn_outputs,self.final_state = self.model()
	
	#计算输出层的输出
	logits = tf.reshape(tf.matmul(tf.reshape(self.rnn_outputs,[-1,self.state_size]),self.W)+self.b,[self.num_batch,self.time_step,2])
	
	self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=logits)
	self.total_loss = tf.reduce_mean(self.losses)
	
	def model(self):
	    cell= rnn.BasicRNNCell(self.state_size)
		rnn_outputs,final_state = tf.nn.dynamic_rnn(cell,self.rnn_inputs,init_state=self.init_state)
		return rnn_outputs,final_state
		#这里我们使用了“dynamic_rnn”，因此每次会同时处理所有batch的第一组数据，总共处理的次数为：batch_size / time_step。
	
	def train(self):
	    with tf.Session() as sess:
		    sess.run(tf.global_variables_initializer())
			training_losses = []
			d = Data(self.data_size,self.num_batch,self.batch_size,self.time_step)
			training_loss = 0
			training_state = np.zeros((self.num_batch,self.state_size))
			for step,(X,Y) in enumerate(d.generate_epochs):
			    tr_losses,training_loss_,training_state,_=sess.run([self.losses,self.total_loss,self.final_state,self.training_step],feed_dict={self.x:X,self.y:Y,self.init_state:training_state})

                training_loss += training_loss_
				
				
	if __name__ == '__main__':
	    data_size =500000
		batch_size = 2000
		time_step = 5
		state_size = 6
		
		m = Model(data_size,batch_size,time_step,state_size)
		training_losses = m.train()
		plt.plot(training_losses)
		plt.show()