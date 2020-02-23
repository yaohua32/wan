import os
os.environ["CUDA_DEVICES_ORDER"]= "PCI_BUS_IS"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

class pde_wan():
    
    def __init__(self, dim, beta, N_int, N_bd):
        import numpy as np
        global np
        #
        import time
        global time
        #
        import tensorflow as tf
        global tf
        #
        import matplotlib.pyplot as plt
        global plt
        #
        from scipy.interpolate import griddata
        global griddata
        #
        self.up=   1.0
        self.low= -1.0
        self.la1= np.pi/2
        self.la2= 1.0/2
        self.mesh_size= 50
        self.beta= beta
        #
        self.v_layer= 6
        self.v_hidden_size= 50
        self.v_step=  1       
        self.v_rate=  0.04   
        self.u_layer= 6
        self.u_hidden_size= 20
        self.u_step= 2 
        self.u_rate= 0.015
        self.batch_size= N_int
        self.test_size=  5000
        self.bound_size= N_bd  #(*2*dim)
        self.iteration=  20001 
        self.dim= dim 
        
    def sample_train(self, dm_size, bd_size, dim):
        low, up, la1, la2= self.low, self.up, self.la1, self.la2
        #*********************************************************
        # collocation points in domain
        x_dm= np.random.uniform(low, up, [dm_size, dim])
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
        #*********************************************************
        int_dm= (up-low)**dim        
        #********************************************************
        # Value of f(x)
        x1_pow= np.power(x_dm[:,0], 2)
        x2_pow= np.power(x_dm[:,1], 2)
        x_y= np.sum(np.power(x_dm, 2), 1)
        #
        f_dm= (4*(x1_pow*la1**2+x2_pow*la2**2)*(x_y+1)*np.sin(la1*x1_pow+la2*x2_pow)
                -(4*la1*x1_pow+4*la2*x2_pow+2*(la1+la2)*(1+x_y))*np.cos(la1*x1_pow+la2*x2_pow)
                +2*(x1_pow*la1**2+x2_pow*la2**2)*np.cos(la1*x1_pow+la2*x2_pow)**2)
        f_dm= np.reshape(f_dm, [-1, 1])
        #*********************************************************
        # Dirichlet boundary condition
        x1_pow= np.power(x_bd[:,0], 2)
        x2_pow= np.power(x_bd[:,1], 2)
        u_bd= np.sin(la1*x1_pow+la2*x2_pow)
        u_bd= np.reshape(u_bd, [-1, 1])
        #*********************************************************
        x_dm= np.float32(x_dm)
        x_bd= np.float32(x_bd)
        int_dm= np.float32(int_dm)
        f_dm= np.float32(f_dm)
        u_bd= np.float32(u_bd)
        return(x_dm, f_dm, x_bd, u_bd, int_dm)

    def sample_test(self, test_size, dim):
        low, up, la1, la2 = self.low, self.up, self.la1, self.la2
        #**********************************************************
        x_mesh= np.linspace(low, up, self.mesh_size)
        x_dm= np.random.uniform(low, up, [test_size, dim])
        #***********************************************************
        # Value of u(x)
        x1_pow= np.power(x_dm[:,0], 2)
        x2_pow= np.power(x_dm[:,1], 2)
        u_dm= np.sin(la1*x1_pow+la2*x2_pow)
        u_dm= np.reshape(u_dm, [-1, 1])
        #***********************************************************
        x_dm= np.float32(x_dm)
        u_dm= np.float32(u_dm)
        mesh= np.meshgrid(x_mesh, x_mesh)
        return(mesh, x_dm, u_dm)
    
    def DNN_u(self, x_in, out_size, name, reuse):
        h_size= self.u_hidden_size
        with tf.variable_scope(name, reuse= reuse):
            hi= tf.layers.dense(x_in, h_size, activation= tf.nn.tanh,
                                name='input_layer')
            hi= tf.layers.dense(hi, h_size, activation= tf.nn.tanh,
                                name='input_layer1')
            for i in range(self.u_layer):
                if i%2==0:
                    hi= tf.layers.dense(hi, h_size, activation= tf.nn.softplus,
                                        name='h_layer'+str(i))
                else:
                    hi= tf.sin(tf.layers.dense(hi, h_size), name='h_layer'+str(i))
            out= tf.layers.dense(hi, out_size, name='out_layer')
        return(out)
    
    def DNN_v(self, x_in, out_size, name, reuse):
        h_size= self.v_hidden_size
        with tf.variable_scope(name, reuse= reuse):
            hi= tf.layers.dense(x_in, h_size, activation= tf.nn.tanh,
                                name='input_layer')
            hi= tf.layers.dense(hi, h_size, activation= tf.nn.tanh,
                                name='input_layer1')
            for i in range(self.v_layer):
                if i%2==0:
                    hi= tf.layers.dense(hi, h_size, activation= tf.nn.softplus,
                                                            name='h_layer'+str(i))
                else:
                    hi= tf.sin(tf.layers.dense(hi, h_size), name='h_layer'+str(i))
            out= tf.layers.dense(hi, out_size, name='out_layer')
        return(out)
    
    def grad_u(self, x, out_size, name):
        #**************************************
        # u(x,y)
        fun_u= self.DNN_u(x, out_size, name, tf.AUTO_REUSE)
        #*************************************
        # grad_u(x,y)
        grad_u= tf.gradients(fun_u, x, unconnected_gradients='zero')[0]
        return(fun_u, grad_u)
    
    def grad_v(self, x, out_size, name):
        #**************************************
        # v(x,y)
        fun_v= self.DNN_v(x, out_size, name, tf.AUTO_REUSE)
        #*************************************
        # grad_v(x,y)
        grad_v= tf.gradients(fun_v, x, unconnected_gradients='zero')[0]
        return(fun_v, grad_v)
    
    def fun_a(self, x):
        #********************************************************
        a_val= tf.add(1.0, tf.reduce_sum(tf.pow(x, 2), 1))
        out= tf.reshape(a_val, [-1,1])
        return(out)
    
    def fun_w(self, x, low=-1.0, up=1.0):
        I1= 0.210987
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
    
    def build(self):
        #**************************************************************
        with tf.name_scope('placeholder'):
            self.x_domain= tf.placeholder(tf.float32, shape=[None, self.dim], name='x_dm')
            self.f_obv= tf.placeholder(tf.float32, shape=[None, 1], name='f_obv')
            self.x_bound= tf.placeholder(tf.float32, shape=[None, self.dim], name='x_b')
            self.g_obv= tf.placeholder(tf.float32, shape=[None, 1], name='g_obv')
            self.int_domain= tf.placeholder(tf.float32, shape=(), name='int_domain')
        #**************************************************************
        name_u= 'dnn_u'; name_v= 'dnn_v';
        self.u_val, self.du= self.grad_u(self.x_domain, 1, name_u)
        self.v_val, self.dv= self.grad_v(self.x_domain, 1, name_v)
        self.w_val, self.dw= self.fun_w(self.x_domain)
        u_bound= self.DNN_u(self.x_bound, 1, name_u, tf.AUTO_REUSE)
        #
        a_val= self.fun_a(self.x_domain)
        self.wv= tf.multiply(self.w_val, self.v_val)
        #
        du_du= tf.reduce_sum(tf.multiply(self.du, self.du), axis=1)
        du_du= tf.reshape(du_du, [-1,1])
        #
        du_dw= tf.reduce_sum(tf.multiply(self.du, self.dw), axis=1)
        du_dw= tf.reshape(du_dw, [-1,1])
        #
        du_dv= tf.reduce_sum(tf.multiply(self.du, self.dv), axis=1)
        du_dv= tf.reshape(du_dv, [-1,1])
        #
        du_dwv= tf.add(tf.multiply(self.v_val, du_dw),
                       tf.multiply(self.w_val, du_dv))
        #**************************************************************
        with tf.name_scope('loss'):
            with tf.name_scope('u_loss'):
                test_norm = tf.multiply(tf.reduce_mean(self.wv**2), self.int_domain)
                #
                int_l1= tf.multiply(tf.reduce_mean(
                        tf.multiply(a_val, du_dwv)), self.int_domain)
                int_l2= tf.multiply(tf.reduce_mean(tf.multiply(self.wv, 0.5*du_du)),self.int_domain)
                int_r= tf.multiply(tf.reduce_mean(
                        tf.multiply(self.f_obv, self.wv)), self.int_domain)
                #
                self.loss_int= tf.square(int_l1+int_l2-int_r) / test_norm                       
                # 
                self.loss_bound= tf.reduce_mean(tf.abs(u_bound-self.g_obv))     
                #
                self.loss_u= (self.beta)*self.loss_bound+(1.0)*self.loss_int
            with tf.name_scope('v_loss'):
                # 
                self.loss_v= - tf.log(self.loss_int)
        #**************************************************************
        # 
        u_vars= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dnn_u')
        v_vars= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dnn_v')
        #***************************************************************
        # 
        with tf.name_scope('optimizer'):
            self.u_opt= tf.train.AdagradOptimizer(self.u_rate).minimize(
                    self.loss_u, var_list= u_vars)
            self.v_opt= tf.train.AdagradOptimizer(self.v_rate).minimize(
                    self.loss_v, var_list= v_vars)
    
    def train(self):
        #****************************************************************
        tf.reset_default_graph(); self.build(); 
        #***************************************************************
        # testing data
        mesh, test_x, test_u= self.sample_test(self.test_size, self.dim);
        step=[]; error_l2r=[]; error_l2=[];
        time_start= time.time()
        #***************************************************************
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #****************************
            for i in range(self.iteration):
                # training data (mini-batch)
                train_data= self.sample_train(self.batch_size, self.bound_size, self.dim)
                feed_train={self.x_domain: train_data[0],
                            self.f_obv: train_data[1],
                            self.x_bound: train_data[2],
                            self.g_obv: train_data[3],
                            self.int_domain: train_data[4]}
                if i%5==0:
                    #*********************************
                    pred_u, pred_v= sess.run([self.u_val, self.v_val],feed_dict={self.x_domain: test_x})
                    err_l2= np.sqrt(np.mean(np.square(test_u-pred_u)))
                    u_norm= np.sqrt(np.mean(np.square(test_u)))
                    step.append(i+1); 
                    error_l2r.append(err_l2/u_norm); error_l2.append(err_l2)          
                if i%200==0:
                    loss_v, loss_u, int_u, loss_bd= sess.run(
                        [self.loss_v, self.loss_u, self.loss_int, self.loss_bound], 
                        feed_dict= feed_train)
                    print('Iterations:{}'.format(i))
                    print('u_loss:{} v_loss:{}'.format(loss_u, loss_v))
                    print('int_u:{} loss_bd:{} err_l2r:{}'.format(
                        int_u, loss_bd, error_l2r[-1]))
                #
                for _ in range(self.v_step):
                    _ = sess.run(self.v_opt, feed_dict=feed_train)
                for _ in range(self.u_step):
                    _ = sess.run(self.u_opt, feed_dict=feed_train)
                #
            time_end= time.time()
            #*******************************************
            print('L2_e is {}, L2_re is {}'.format(error_l2[-1], error_l2r[-1]))
            print('Total running time is {}:'.format(time_end-time_start))
        return(mesh, test_x, test_u, pred_u, step, error_l2, error_l2r, self.dim)

if __name__=='__main__':
    dim, beta, N_int, N_bd= 5,  20000000, 20000, 100
    demo= pde_wan(dim, beta, N_int, N_bd)
    mesh, test_x, test_u, pred_u, step, error_l2, error_l2r, dim= demo.train()
    # save data as .mat form
    import scipy.io
    data_save= {}
    data_save['mesh']= mesh
    data_save['test_x']= test_x
    data_save['test_u']= test_u
    data_save['pred_u']= pred_u
    data_save['step']= step
    data_save['error_l2']= error_l2
    data_save['error_l2r']= error_l2r
    scipy.io.savemat('WAN_nonlinear_%dd'%(dim), data_save)
