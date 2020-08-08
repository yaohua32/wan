class wan_pde_solver():
    
    def __init__(self, dim, N_dm, N_bd, file_path, beta_int, beta_intw, beta_bd,
                 v_step, v_rate, u_step, u_rate, u_layer=6, u_neuron=40,
                 iteration=20001):
        import numpy as np
        global np
        #
        import time
        global time
        #
        import tensorflow.compat.v1 as tf
        global tf
        tf.disable_v2_behavior()
        #
        import matplotlib.pyplot as plt
        global plt
        #
        from scipy.interpolate import griddata
        global griddata
        #
        from scipy.stats import truncnorm
        global truncnorm
        # 
        from matplotlib import cm
        global cm
        #
        self.low, self.up= 0.0, 1.0
        self.dim= dim                           #dimension of the problem
        self.mesh_size= 50                      #for generating testing data
        self.dm_size= N_dm                      #collocation points inside domain                   
        self.bd_size= N_bd                      #collocation points on boundary
        self.iteration= iteration
        self.dir= file_path
        self.beta_int= beta_int
        self.beta_intw= beta_intw
        self.beta_bd= beta_bd
        #
        self.v_layer= 2                          
        self.v_h_size= 40                        
        self.v_step= v_step                         
        self.v_rate= v_rate   #0.01
        #
        self.u_layer= u_layer                          
        self.u_h_size= u_neuron                        
        self.u_step= u_step                    #u(x)inner loop（iteration for u）
        self.u_rate= u_rate                    #u(x)inner loop（learning rate for u）
        
    def sample_train (self, dm_size, bd_size, dim):
        low, up= self.low, self.up
        # generate taining data
        #********************************************************
        # collocation points in domain
        x_dm= np.random.uniform(low, up, [dm_size, dim])
        #*********************************************************
        # The value of f(x)
        x= np.reshape(x_dm[:,0], [-1,1])
        f_dm= -2*np.ones([np.shape(x_dm)[0], 1])
        #*********************************************************
        # collocation points on boundary
        x_bd_list=[]
        for i in range(dim):
            x_bound= np.random.uniform(low, up, [bd_size, dim])
            x_bound[:,i]= up
            x_bd_list.append(x_bound)
            x_bound= np.random.uniform(low, up, [bd_size, dim])
            x_bound[:,i]= low
            x_bd_list.append(x_bound)
        x_bd= np.concatenate(x_bd_list, axis=0)     
        #***********************************************************
        # observation of u(x) on boundary
        x= np.reshape(x_bd[:,0], [-1,1])
        u_bd= np.where(x<=0.5, x**2, (x-1)**2)
        #
        train_dict={}
        x_dm= np.float32(x_dm); train_dict['x_dm']= x_dm
        x_bd= np.float32(x_bd); train_dict['x_bd']= x_bd
        f_dm= np.float32(f_dm); train_dict['f_val']= f_dm
        u_bd= np.float32(u_bd); train_dict['u_bd']= u_bd
        return(train_dict)
        
    def sample_test(self, mesh_size, dim):
        # testing data
        low, up= self.low, self.up
        #**********************************************************
        # generate meshgrid in the domain
        x_mesh= np.linspace(low, up, mesh_size)
        mesh= np.meshgrid(x_mesh, x_mesh)
        #
        x1_dm= np.reshape(mesh[0], [-1,1])
        x2_dm= np.reshape(mesh[1], [-1,1])
        #
        x3_dm= np.random.uniform(low, up, [mesh_size*mesh_size, dim-2])
        x_dm= np.concatenate([x1_dm, x2_dm, x3_dm], axis=1)
        x4_dm= np.zeros([mesh_size*mesh_size, dim-2])
        x_draw_dm= np.concatenate([x1_dm, x2_dm, x4_dm], axis=1)
        #***********************************************************
        # The exact u(x)
        u_dm= np.where(x1_dm<=0.5, x1_dm**2, (x1_dm-1)**2)
        #
        u_draw_dm= np.where(x1_dm<=0.5, x1_dm**2, (x1_dm-1)**2)
        #***********************************************************
        test_dict={}; test_dict['mesh']= mesh
        x_dm= np.float32(x_dm); test_dict['test_x']= x_dm
        x_draw_dm= np.float32(x_draw_dm); test_dict['draw_x']= x_draw_dm 
        u_dm= np.float32(u_dm); test_dict['test_u']= u_dm 
        u_draw_dm= np.float32(u_draw_dm); test_dict['draw_u']= u_draw_dm
        return(test_dict)

    def net_u(self, x_in, out_size, name, reuse):
        h_size= self.u_h_size
        with tf.variable_scope(name, reuse= reuse):
            hi= tf.layers.dense(x_in, h_size, activation= tf.nn.tanh,
                                name='input_layer')
            for i in range(self.u_layer):
                if i%2==0:
                    hi= tf.layers.dense(hi, h_size, activation= tf.nn.tanh,
                                        name='h_layer'+str(i))
                else:
                    hi= tf.layers.dense(hi, h_size, activation= tf.nn.elu,
                                        name='h_layer'+str(i))
            out= tf.layers.dense(hi, out_size, name='out_layer')
        return(out)


    def net_v(self, x_in, out_size, name, reuse):
        h_size= self.v_h_size
        with tf.variable_scope(name, reuse= reuse):
            hi= tf.layers.dense(x_in, h_size, activation= tf.nn.tanh,
                                name='input_layer')
            hi= tf.layers.dense(hi, h_size, activation= tf.nn.softplus,
                                name='input_layer1')
            for i in range(self.v_layer):
                if i%2==0:
                    hi= tf.layers.dense(hi, h_size, activation= tf.nn.softplus,
                                                            name='h_layer'+str(i))
                else:
                    hi= tf.sin(tf.layers.dense(hi, h_size), name='h_layer'+str(i))
            out= tf.layers.dense(hi, out_size, name='out_layer')
        return(out)

    def fun_w(self, x, low, up):
        I1= 0.110987
        x_list= tf.split(x, self.dim, 1)
        #**************************************************
        x_scale_list=[]
        h_len= (up-low)/2.0
        for i in range(self.dim):
            x_scale= (x_list[i]-low-h_len)/h_len
            x_scale_list.append(x_scale)
        #************************************************
        z_x_list=[];
        for i in range(self.dim):
            supp_x= tf.greater(1-tf.abs(x_scale_list[i]), 0)
            z_x= tf.where(supp_x, tf.exp(1/(tf.pow(x_scale_list[i], 2)-1))/I1, 
                          tf.zeros_like(x_scale_list[i]))
            z_x_list.append(z_x)
        #***************************************************
        w_val= tf.constant(1.0)
        for i in range(self.dim):
            w_val= tf.multiply(w_val, z_x_list[i])
        dw= tf.gradients(w_val, x, unconnected_gradients='zero')[0]
        dw= tf.where(tf.is_nan(dw), tf.zeros_like(dw), dw)
        return(w_val, dw)
    
    def grad_u(self, x_in, name, out_size=1):
        x_list= tf.split(x_in, self.dim, axis=1)
        x_in= tf.concat(x_list, axis=1)
        #****************************************************
        u_val= self.net_u(x_in, out_size, name, tf.AUTO_REUSE)
        #
        grad_u= tf.gradients(u_val, x_in, unconnected_gradients='zero')[0]
        #*****************************************************
        grad_u_list= tf.split(grad_u, self.dim, axis=1)
        #
        div_du= tf.zeros_like(grad_u_list[0])
        for i in range(self.dim):
            div_du_i= tf.gradients(grad_u_list[i], x_list[i], unconnected_gradients='zero')[0]
            div_du= div_du+div_du_i
        return(u_val, grad_u, div_du)
        
    def grad_v(self, x_in, name, out_size=1):
        v_val= self.net_v(x_in, out_size, name, tf.AUTO_REUSE)
        #
        grad_v= tf.gradients(v_val, x_in, unconnected_gradients='zero')[0]
        return(v_val, grad_v)
    
    def build(self):
        #*********************************************************************
        with tf.name_scope('placeholder'):
            self.x_dm= tf.placeholder(tf.float32, shape=[None, self.dim], name='x_dm')
            self.x_bd= tf.placeholder(tf.float32, shape=[None, self.dim], name='x_bd')
            self.f_val= tf.placeholder(tf.float32, shape=[None, 1], name='f_val')
            self.u_bd= tf.placeholder(tf.float32, shape=[None, 1], name='u_bd')
        #*********************************************************************
        name_v='net_v'; name_u='net_u';
        self.u_val, grad_u, self.div_du= self.grad_u(self.x_dm, name_u)
        self.v_val, grad_v= self.grad_v(self.x_dm, name_v)
        w_val, grad_w= self.fun_w(self.x_dm, self.low, self.up)
        #
        u_bd_pred, _, _= self.grad_u(self.x_bd, name_u)
        #**********************************************************************
        self.wv_val= tf.multiply(w_val, self.v_val)
        #
        dudw_val= tf.reduce_sum(tf.multiply(grad_u, grad_w), axis=1)
        dudw_val= tf.reshape(dudw_val, [-1,1])
        #
        dudv_val= tf.reduce_sum(tf.multiply(grad_u, grad_v), axis=1)
        dudv_val= tf.reshape(dudv_val, [-1,1])
        #
        dudwv_val= tf.add(tf.multiply(self.v_val, dudw_val),
                          tf.multiply(w_val, dudv_val))
        #*****************************************************************
        with tf.variable_scope('loss'):
            with tf.name_scope('loss_u'):
                test_norm= tf.reduce_mean(self.wv_val**2)
                w_norm= tf.reduce_mean(w_val**2) 
                #*********************************************************
                point_dist= dudwv_val-tf.multiply(self.f_val, self.wv_val)
                int_1= tf.reduce_mean(point_dist)
                self.loss_int= self.beta_int*tf.square(int_1) / test_norm
                #
                point_dist= dudw_val-tf.multiply(self.f_val, w_val)
                int_1= tf.reduce_mean(point_dist)
                self.loss_intw= self.beta_intw*tf.square(int_1)/ w_norm
                #**********************************************************
                self.loss_bd= tf.reduce_mean(tf.abs(u_bd_pred-self.u_bd))
                #
                self.loss_u= (self.beta_bd)*(self.loss_bd)+self.beta_int*self.loss_int
            with tf.name_scope('loss_v'):
                # 
                self.loss_v=  - tf.log(self.loss_int)
        #**************************************************************
        # 
        u_vars= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name_u)
        v_vars= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name_v)
        #***************************************************************
        # 
        with tf.name_scope('optimizer'):
            #self.u_opt= tf.train.AdagradOptimizer(self.u_rate).minimize(
            #        self.loss_u, var_list= u_vars)
            self.u_opt= tf.train.AdamOptimizer(self.u_rate).minimize(
                    self.loss_u, var_list= u_vars)
            self.v_opt= tf.train.AdagradOptimizer(self.v_rate).minimize(
                    self.loss_v, var_list= v_vars)
    
    def main_fun(self):
        #*********************************************************************
        tf.reset_default_graph(); self.build()
        #*********************************************************************
        # generate points for testing usage
        test_dict= self.sample_test(self.mesh_size, self.dim)
        #saver= tf.train.Saver()
        list_dict={}
        step_list=[]; err_l2r_list=[]; err_l1_list=[]; train_loss_list=[]
        sample_time=[]; train_time=[]
        #
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.iteration):
                ################################################sampling step
                sample_time0= time.time()
                train_dict= self.sample_train(self.dm_size, self.bd_size, self.dim)
                feed_train= {self.x_dm: train_dict['x_dm'],
                             self.f_val: train_dict['f_val'],
                            self.x_bd: train_dict['x_bd'],
                            self.u_bd: train_dict['u_bd']}
                sample_time.append(time.time()-sample_time0)
                #
                if i%5==0:
                    pred_u, div_du= sess.run([self.u_val, self.div_du],feed_dict={self.x_dm: test_dict['test_x']}) 
                    #
                    err_l2= np.sqrt(np.mean(np.square(test_dict['test_u']-pred_u)))
                    u_l2= np.sqrt(np.mean(np.square(test_dict['test_u'])))
                    err_l2r_list.append(err_l2/u_l2)
                    #
                    err_l1= np.mean(np.abs(test_dict['test_u']-pred_u))
                    err_l1_list.append(err_l1)
                    #
                    step_list.append(i+1)
                    #
                    loss_u, loss_v, loss_int, loss_bd= sess.run(
                        [self.loss_u, self.loss_v, self.loss_int, self.loss_bd], 
                        feed_dict= feed_train)
                    train_loss_list.append(loss_u)
                    #
                    if i%500==0:
                        print('Iterations:{}'.format(i))
                        print('loss_u:{} loss_v:{} loss_int:{} loss_bd:{} l2r:{}'.format(
                            loss_u, loss_v, loss_int, loss_bd, err_l2r_list[-1]))
                        #
                        pred_u_draw, pred_v_draw= sess.run(
                                [self.u_val, self.v_val], 
                                feed_dict={self.x_dm: test_dict['draw_x']})
                ##################################################training step
                train_time0= time.time()
                for _ in range(self.v_step):
                    _ = sess.run(self.v_opt, feed_dict=feed_train)                    
                for _ in range(self.u_step):
                    _ = sess.run(self.u_opt, feed_dict=feed_train)
                train_time.append(time.time()-train_time0)
                #
            #*****************************************************************
            print('Running time is:{}'.format(sum(train_time)+sum(sample_time)))
            #
            list_dict['err_l2r_list']= err_l2r_list
            list_dict['err_l1_list']= err_l1_list
            list_dict['train_loss_list']= train_loss_list
            list_dict['step_list']= step_list
            list_dict['sample_time']= sample_time
            list_dict['train_time']= train_time
        return(test_dict, pred_u, div_du, pred_u_draw, list_dict, self.dim)


layer= 4; neuron= 40
N_dm, N_bd= 10000, 100
print('****************layer:{} neuron:{}***************'.format(2*layer+2, neuron))
dim, iteration=2, 20001
file_path= './problem_sigl/'
beta_int, beta_intw, beta_bd= 1.0, 0.0, 1000000.0
v_step, v_rate, u_step, u_rate= 1, 0.015, 1, 0.001
#****************************
demo= wan_pde_solver(dim, N_dm, N_bd, file_path, beta_int, beta_intw, beta_bd, 
                    v_step, v_rate, u_step, u_rate, layer, neuron, iteration)
test_dict, pred_u, pred_div_du, pred_u_draw, list_dict, dim= demo.main_fun()
#***************************
# save data as .mat form
import scipy.io
data_save= {}
data_save['test_dict']= test_dict
data_save['pred_u']= pred_u
data_save['pred_div_du']= pred_div_du
data_save['pred_u_draw']= pred_u_draw
data_save['list_dict']= list_dict
scipy.io.savemat(file_path+'WAN_pde_%dd'%(dim), data_save)
print('Data saved in '+file_path)















