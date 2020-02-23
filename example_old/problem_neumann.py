import os
os.environ["CUDA_DEVICES_ORDER"]= "PCI_BUS_IS"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

class pde_wan():
    
    def __init__(self, dim=10, beta=1000):
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
        self.up= 1.0
        self.low= 0.0
        self.la= np.pi/2
        self.mesh_size= 50
        self.beta= beta
        #
        self.v_layer= 6
        self.v_hidden_size= 50
        self.v_step= 2    
        self.v_rate=0.05
        self.u_layer= 6
        self.u_hidden_size= 20
        self.u_step= 5
        self.u_rate=0.02
        self.batch_size= 80000
        self.test_size=  5000
        self.bound_size= 400 
        self.iteration=  20001 
        self.dim= dim 
        self.dir= './problem_neumann/'   
        
    def sample_train(self, dm_size, bd_size, dim):
        '''
        - Laplace U + 2U = f
        u(x)= sin(la*x1)*cos(la*x2)    
        f(x)= 2*(la^2+1)*sin(la*x1)*cos(la*x2)
        '''
        low= self.low; up= self.up; la= self.la
        #*********************************************************
        # collocation points in domain
        x_dm= np.random.uniform(low, up, [dm_size, dim])
        #**************************************
        # collocation points on the boundary
        x_bd_list=[]
        n_vector_list=[]
        for i in range(dim):
            x_bd_data= np.random.uniform(low, up, [bd_size, dim])
            x_bd_data[:,i]= up
            x_bd_list.append(x_bd_data)
            n_vector= np.zeros_like(x_bd_data)
            n_vector[:,i]=1
            n_vector_list.append(n_vector)
            x_bd_data= np.random.uniform(low, up, [bd_size, dim])
            x_bd_data[:,i]= low
            x_bd_list.append(x_bd_data)
            n_vector= np.zeros_like(x_bd_data)
            n_vector[:,i]=-1
            n_vector_list.append(n_vector)
        x_bd= np.concatenate(x_bd_list, 0)
        n_vector= np.concatenate(n_vector_list, 0)
        #*********************************************************
        int_dm= (up-low)**dim; int_surf= 2*dim*(up-low)**(dim-1)    
        #********************************************************
        f_dm= 2*(la**2+1)*np.sin(la*x_dm[:,0])*np.cos(la*x_dm[:,1])
        f_dm= np.reshape(f_dm, [-1, 1])
        #*********************************************************
        x_dm= np.float32(x_dm)
        f_dm= np.float32(f_dm)
        x_bd= np.float32(x_bd)
        n_vector= np.float32(n_vector)
        int_dm= np.float32(int_dm)
        int_surf= np.float32(int_surf)
        return(x_dm, f_dm, x_bd, n_vector, int_dm, int_surf)

    def sample_test(self, test_size, dim):
        up= self.up; low= self.low; la= self.la
        #**********************************************************
        x_mesh= np.linspace(low, up, self.mesh_size)
        x_dm= np.random.uniform(low, up, [test_size, dim])
        #***********************************************************
        u_dm= np.sin(la*x_dm[:,0])*np.cos(la*x_dm[:,1])
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

    def fun_g(self, x, n_vec):
        x_list= tf.split(x, self.dim, 1)
        la= self.la
        #**************************************
        u= tf.multiply(tf.sin(la*x_list[0]), tf.cos(la*x_list[1]))
        #
        du= tf.gradients(u, x, unconnected_gradients='zero')[0]
        g_obv= tf.reduce_sum(tf.multiply(du, n_vec), axis=1)
        g_obv= tf.reshape(g_obv, [-1,1])
        return(u, g_obv)
        
    def build(self):
        #**************************************************************
        with tf.name_scope('placeholder'):
            self.x_domain= tf.placeholder(tf.float32, shape=[None, self.dim], name='x_dm')
            self.f_obv= tf.placeholder(tf.float32, shape=[None, 1], name='f_obv')
            self.x_bound= tf.placeholder(tf.float32, shape=[None, self.dim], name='x_b')
            self.n_vec= tf.placeholder(tf.float32, shape=[None, self.dim], name='n_vec')
            self.int_domain= tf.placeholder(tf.float32, shape=(), name='int_domain')
            self.int_surf= tf.placeholder(tf.float32, shape=(), name='int_surface')
        #**************************************************************
        name_u= 'dnn_u'; name_v= 'dnn_v';
        self.u_val, self.du= self.grad_u(self.x_domain, 1, name_u)
        u_boound, du_bd= self.grad_u(self.x_bound, 1, name_u)
        self.v_val, self.dv= self.grad_v(self.x_domain, 1, name_v)
        v_bd, _ = self.grad_v(self.x_bound, 1, name_v)
        self.w_val, self.dw= self.fun_w(self.x_domain)
        #
        self.wv= tf.multiply(self.w_val, self.v_val)
        #
        du_dw= tf.reduce_sum(tf.multiply(self.du, self.dw), axis=1)
        du_dw= tf.reshape(du_dw, [-1,1])
        #
        du_dv= tf.reduce_sum(tf.multiply(self.du, self.dv), axis=1)
        du_dv= tf.reshape(du_dv, [-1,1])
        #
        u_v= tf.multiply(self.u_val, self.v_val)
        #
        f_v= tf.multiply(self.f_obv, self.v_val)
        #
        u_obv, g_obv= self.fun_g(self.x_bound, self.n_vec)
        #
        v_g_bd= tf.multiply(v_bd, g_obv)
        #**************************************************************
        with tf.name_scope('loss'):
            with tf.name_scope('u_loss'):
                v_norm= tf.multiply(tf.reduce_mean(self.v_val**2), self.int_domain)
                # weak form (v)
                int_l1= tf.multiply(tf.reduce_mean(du_dv), self.int_domain)
                int_l2= tf.multiply(tf.reduce_mean(tf.multiply(2.0, u_v)), self.int_domain)
                int_r1= tf.multiply(tf.reduce_mean(f_v), self.int_domain)
                int_r2= tf.multiply(tf.reduce_mean(v_g_bd), self.int_surf)
                # compatiability condition
                int_l_u= tf.multiply(tf.reduce_mean(tf.multiply(self.u_val, 2.0)), self.int_domain)
                int_r1_u= tf.multiply(tf.reduce_mean(self.f_obv), self.int_domain)
                int_r2_u= tf.multiply(tf.reduce_mean(g_obv), self.int_surf)
                #
                self.loss_int= tf.square(int_l1+int_l2-int_r1-int_r2) / v_norm
                #
                self.comp= tf.square(int_l_u-int_r1_u-int_r2_u)
                # 
                g_val= tf.reduce_sum(tf.multiply(du_bd, self.n_vec), axis=1)
                g_val= tf.reshape(g_val, [-1,1]) 
                self.loss_bound= tf.reduce_mean(tf.abs(g_val -g_obv))
                #
                self.loss_u= (1.0)*self.loss_int+(self.beta)*self.loss_bound
            with tf.name_scope('v_loss'):
                self.loss_v= - (1.0)*tf.log(self.loss_int)
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
        mesh, test_x, test_u= self.sample_test(self.test_size, self.dim);
        step=[]; error_l2r=[]; error_l2=[]; 
        time_begin=time.time(); time_list=[]; iter_time_list=[]
        #***************************************************************
        #saver= tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #****************************
            for i in range(self.iteration):
                # mini-batch data
                train_data= self.sample_train(self.batch_size, self.bound_size, self.dim)
                feed_train={self.x_domain: train_data[0],
                            self.f_obv: train_data[1],
                            self.x_bound: train_data[2],
                            self.n_vec: train_data[3],
                            self.int_domain: train_data[4],
                            self.int_surf: train_data[5]}
                if i%5==0:
                    #*********************************
                    pred_u, pred_v= sess.run([self.u_val, self.v_val],feed_dict={self.x_domain: test_x})
                    err_l2= np.sqrt(np.mean(np.square(test_u-pred_u)))
                    u_norm= np.sqrt(np.mean(np.square(test_u)))
                    step.append(i+1); 
                    error_l2r.append(err_l2/u_norm); error_l2.append(err_l2)
                    time_step= time.time(); time_list.append(time_step-time_begin)            
                if i%200==0:
                    loss_v, loss_u, int_u, loss_bd= sess.run(
                        [self.loss_v, self.loss_u, self.loss_int, self.loss_bound], 
                        feed_dict= feed_train)
                    print('Iterations:{}'.format(i))
                    print('u_loss:{} v_loss:{}'.format(loss_u, loss_v))
                    print('int_u:{} loss_bd:{} err_l2r:{}'.format(
                        int_u, loss_bd, error_l2r[-1]))
                #
                iter_time0= time.time()
                for _ in range(self.v_step):
                    _ = sess.run(self.v_opt, feed_dict=feed_train)
                for _ in range(self.u_step):
                    _ = sess.run(self.u_opt, feed_dict=feed_train)
                iter_time_list.append(time.time()-iter_time0)
                #
            #*******************************************
            print('L2_e is {}, L2_re is {}'.format(np.min(error_l2), np.min(error_l2r)))
        return(mesh, test_x, test_u, pred_u, step, error_l2, error_l2r, time_list, iter_time_list, self.dim)

if __name__=='__main__':
    demo= pde_wan()
    mesh, test_x, test_u, pred_u, step, error_l2, error_l2r, time_list, iter_time_list, dim= demo.train()
    #***************************
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
    data_save['time_list']= time_list
    data_save['iter_time_list']= iter_time_list
    scipy.io.savemat('./problem_neumann/'+'wan_pde_%dd'%(dim), data_save)     
        
        
        
        
        
        
    