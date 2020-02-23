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
        self.up= 1.0
        self.low= -1.0
        self.t0= 0.0
        self.t1= 1.0
        self.la= np.pi/2
        self.pho= 2.0        
        self.mu= self.la**2-1
        self.mesh_size= 50
        self.beta= beta
        #
        self.v_layer= 6
        self.v_hidden_size= 50
        self.v_step= 1       
        self.v_rate=0.04   
        self.u_layer= 6
        self.u_hidden_size= 20
        self.u_step= 2 
        self.u_rate=0.015
        self.batch_size= N_int
        self.test_size=  5000
        self.bound_size= N_bd  
        self.iteration=  20001 
        self.dim= dim 
        self.dir= './problem_parabolic/'
        
    def sample_train(self, inside_size, bound_size, dim=2):
        '''
        u(x)= pho*sin(la*x1)*exp{(mu-la^2)*t}
        Thus, g(x)= pho*sin(la*x1) 
        '''
        la, low, up, t0, t1= self.la, self.low, self.up, self.t0, self.t1
        # colloaction points in the domain
        x_inside= np.random.uniform(low, up, [inside_size, dim])
        t_inside= np.random.uniform(t0, t1, [inside_size, 1])
        x_inside= np.concatenate((x_inside, t_inside), axis=1)
        # value of f
        u_inside= self.pho*np.sin(la*x_inside[:,0])*np.exp((self.mu-la**2)*x_inside[:,-1])
        u_inside= np.reshape(u_inside, [-1,1])
        f_obv= np.multiply(self.mu, u_inside)-np.multiply(u_inside, u_inside)
        #***************************************************
        # initial condition
        x_init= np.random.uniform(low, up, [bound_size, dim+1])
        x_init[:,-1]= t0
        #
        g_obv= self.pho*np.sin(la*x_init[:,0])*np.exp((self.mu-la**2)*x_init[:,-1])
        g_obv= np.reshape(g_obv, [-1,1])
        # end time
        x_right= np.random.uniform(low, up, [bound_size, dim+1])
        x_right[:,-1]= t1
        #****************************************************
        # boundary condition
        x_bound_list=[]
        for i in range(dim):
            x_bound= np.random.uniform(low, up, [bound_size, dim])
            t_bound= np.random.uniform(t0, t1, [bound_size, 1])
            x_bound[:,i]= up
            x_bound_list.append(np.concatenate((x_bound, t_bound), axis=1))
            x_bound= np.random.uniform(low, up, [bound_size, dim])
            t_bound= np.random.uniform(t0, t1, [bound_size, 1])
            x_bound[:,i]= low
            x_bound_list.append(np.concatenate((x_bound, t_bound), axis=1))
        x_bound= np.concatenate(x_bound_list, axis=0)
        # 
        u_bd= self.pho*np.sin(la*x_bound[:,0])*np.exp((self.mu-la**2)*x_bound[:,-1])
        u_bd= np.reshape(u_bd, [-1,1])
        #********************************
        x_inside= np.float32(x_inside)
        f_obv= np.float32(f_obv)
        x_init= np.float32(x_init)
        x_right= np.float32(x_right)
        g_obv= np.float32(g_obv)
        x_bound= np.float32(x_bound)
        u_bd= np.float32(u_bd)
        int_x= np.float32((up-low)**dim)
        int_t= np.float32((t1-t0))
        #**********************************
        return(x_inside, f_obv, x_init, g_obv, x_right, x_bound, u_bd, int_x, int_t)

    def sample_test(self, test_size, dim):
        '''
        u(x)= pho*sin(la*x1)*exp{(mu-la^2)*t}
        Thus, g(x)= pho*sin(la*x1) 
        '''
        la, low, up, t0, t1= self.la, self.low, self.up, self.t0, self.t1
        #*******************************************
        # made data in the domain    
        x_mesh= np.linspace(low, up, self.mesh_size)
        t_mesh= np.linspace(t0, t1, self.mesh_size)
        x_inside= np.random.uniform(self.low, self.up, [test_size, dim])
        t_inside= np.random.uniform(self.t0, self.t1, [test_size, 1])
        x_test= np.concatenate([x_inside, t_inside], axis=1)
        #*******************************************
        # Exact value of u(x,t)
        u_x= self.pho*np.sin(la*x_test[:,0])*np.exp((self.mu-la**2)*x_test[:,-1])
        u_x= np.reshape(u_x, [-1, 1])
        #******************************
        x_test= np.float32(x_test)
        u_x= np.float32(u_x)
        mesh= np.meshgrid(x_mesh, t_mesh)
        return(mesh, x_test, u_x)
    
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
    
    def grad_u(self, x_list, out_size, name):
        x= tf.concat(x_list, axis=1)
        #**************************************
        # u(x,y)
        fun_u= self.DNN_u(x, out_size, name, tf.AUTO_REUSE)
        #*************************************
        # grad_u(x,y)
        grad_u= tf.gradients(fun_u, x_list, unconnected_gradients='zero')
        du_x, du_t = grad_u[0], grad_u[-1]
        return(fun_u, du_x, du_t)
    
    def grad_v(self, x_list, out_size, name):
        x= tf.concat(x_list, axis=1)
        #**************************************
        # v(x,y)
        fun_v= self.DNN_v(x, out_size, name, tf.AUTO_REUSE)
        #*************************************
        # grad_v(x,y)
        grad_v= tf.gradients(fun_v, x_list, unconnected_gradients='zero')
        dv_x, dv_t = grad_v[0], grad_v[-1]
        return(fun_v, dv_x, dv_t)
    
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
            self.x_dm= tf.placeholder(tf.float32, shape=[None, self.dim+1], name='x_dm')
            self.f_obv= tf.placeholder(tf.float32, shape=[None, 1], name='f_obv')
            self.x_init= tf.placeholder(tf.float32, shape=[None, self.dim+1], name='x_init')
            self.g_obv= tf.placeholder(tf.float32, shape=[None, 1], name='g_obv')
            self.x_right= tf.placeholder(tf.float32, shape=[None, self.dim+1], name='x_right')
            self.x_bound= tf.placeholder(tf.float32, shape=[None, self.dim+1], name='x_b')
            self.bd_obv= tf.placeholder(tf.float32, shape=[None, 1], name='bd_obv')
            self.int_x= tf.placeholder(tf.float32, shape=(), name='int_x')
            self.int_t= tf.placeholder(tf.float32, shape=(), name='int_t')
        #**************************************************************
        name_u= 'dnn_u'; name_v= 'dnn_v';
        # initial condition
        x_init_list= tf.split(self.x_init, [self.dim, 1], axis=1)
        u_init= self.DNN_u(self.x_init, 1, name_u, tf.AUTO_REUSE)
        w_init, _= self.fun_w(x_init_list[0])
        v_init= self.DNN_v(self.x_init, 1, name_v, tf.AUTO_REUSE)
        #
        uw_init= tf.multiply(self.g_obv, w_init)
        uwv_init= tf.multiply(uw_init, v_init)
        # end time
        x_right_list= tf.split(self.x_right, [self.dim, 1], axis=1)
        u_right= self.DNN_u(self.x_right, 1, name_u, tf.AUTO_REUSE)
        w_right, _= self.fun_w(x_right_list[0])
        v_right= self.DNN_v(self.x_right, 1, name_v, tf.AUTO_REUSE)
        #
        uw_right= tf.multiply(u_right, w_right)
        uwv_right= tf.multiply(uw_right, v_right)
        # boundary condition
        u_bd= self.DNN_u(self.x_bound, 1, name_u, tf.AUTO_REUSE)
        # inside domain
        x_dm_list= tf.split(self.x_dm, [self.dim, 1], axis=1)
        self.w_dm, dwx_dm= self.fun_w(x_dm_list[0])
        self.v_dm, dvx_dm, dvt_dm= self.grad_v(x_dm_list, 1, name_v)
        self.u_dm, dux_dm, dut_dm= self.grad_u(x_dm_list, 1, name_u)
        self.wv_dm= tf.multiply(self.w_dm, self.v_dm)
        #
        uw_dm= tf.multiply(self.u_dm, self.w_dm)
        uwv_dm= tf.multiply(uw_dm, self.v_dm)
        #
        dux_dw_dm= tf.reduce_sum(tf.multiply(dux_dm, dwx_dm), axis=1)
        dux_dw_dm= tf.reshape(dux_dw_dm, [-1,1])
        dux_dv_dm= tf.reduce_sum(tf.multiply(dux_dm, dvx_dm), axis=1)
        dux_dv_dm= tf.reshape(dux_dv_dm, [-1,1])
        dux_dwv_dm= tf.add(tf.multiply(self.v_dm, dux_dw_dm),
                          tf.multiply(self.w_dm, dux_dv_dm))
        uw_dvt_dm= tf.multiply(uw_dm, dvt_dm)
        #
        f_w_dm= tf.multiply(self.f_obv, self.w_dm)
        f_wv_dm= tf.multiply(self.f_obv, self.wv_dm)
        uu_w_dm= tf.multiply(uw_dm, self.u_dm)
        uu_wv_dm= tf.multiply(uwv_dm, self.u_dm)
        # volume of the whole domain
        int_dm= tf.multiply(self.int_x, self.int_t)
        #**************************************************************
        with tf.name_scope('loss'):
            with tf.name_scope('u_loss'):
                test_norm = tf.multiply(tf.reduce_mean(self.wv_dm**2), int_dm)
                #
                int_l1= tf.multiply(tf.reduce_mean(uwv_right), self.int_x)
                int_l2= tf.multiply(tf.reduce_mean(dux_dwv_dm), int_dm)
                int_r1= tf.multiply(tf.reduce_mean(uwv_init), self.int_x)
                int_r2= tf.multiply(tf.reduce_mean(uu_wv_dm), int_dm)
                int_r3= tf.multiply(tf.reduce_mean(uw_dvt_dm), int_dm)
                int_r4= tf.multiply(tf.reduce_mean(f_wv_dm), int_dm)                
                #
                intw_l1= tf.multiply(tf.reduce_mean(uw_right), self.int_x)
                intw_l2= tf.multiply(tf.reduce_mean(dux_dw_dm), int_dm)
                intw_r1= tf.multiply(tf.reduce_mean(uw_init), self.int_x)
                intw_r2= tf.multiply(tf.reduce_mean(uu_w_dm), int_dm)
                intw_r3= tf.multiply(tf.reduce_mean(f_w_dm), int_dm)
                #
                self.loss_int= tf.square(int_l1+int_l2-int_r1-int_r2-int_r3-int_r4) / test_norm                         
                self.loss_int_w= tf.square(intw_l1+intw_l2-intw_r1-intw_r2-intw_r3)                        
                # 
                self.loss_bound= tf.reduce_sum(tf.abs(u_bd-self.bd_obv))
                self.loss_init= tf.reduce_sum(tf.abs(u_init-self.g_obv)) 
                #
                self.loss_u= self.beta*self.loss_bound+self.beta*self.loss_init+1.0*self.loss_int
            with tf.name_scope('v_loss'):
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
                feed_train={self.x_dm: train_data[0],
                            self.f_obv: train_data[1],
                            self.x_init: train_data[2],
                            self.g_obv: train_data[3],
                            self.x_right: train_data[4],
                            self.x_bound: train_data[5],
                            self.bd_obv: train_data[6],
                            self.int_x: train_data[7],
                            self.int_t: train_data[8]}
                if i%5==0:
                    #*********************************
                    pred_u, pred_v= sess.run([self.u_dm, self.v_dm],feed_dict={self.x_dm: test_x})
                    err_l2= np.sqrt(np.mean(np.square(test_u-pred_u)))
                    u_norm= np.sqrt(np.mean(np.square(test_u)))
                    step.append(i+1); 
                    error_l2r.append(err_l2/u_norm); error_l2.append(err_l2)
                    time_step= time.time(); time_list.append(time_step-time_begin)            
                if i%200==0:
                    loss_v, loss_u, int_u, intw_u, loss_bd= sess.run(
                        [self.loss_v, self.loss_u, self.loss_int, self.loss_int_w, self.loss_bound], 
                        feed_dict= feed_train)
                    print('Iterations:{}'.format(i))
                    print('u_loss:{} v_loss:{}'.format(loss_u, loss_v))
                    print('int_u:{} int_u_w:{} loss_bd:{} err_l2r:{}'.format(
                        int_u, intw_u, loss_bd, error_l2r[-1]))
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
    dim, beta, N_int, N_bd= 10, 20000, 40000, 200
    demo= pde_wan(dim, beta, N_int, N_bd)
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
    scipy.io.savemat('./problem_parabolic/'+'wan_pde_%dd'%(dim), data_save)