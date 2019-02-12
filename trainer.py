
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorlayer as tl


from multiprocessing import Pool

from ETL import put_to_npz, save_npz_dict, assign_params, load_npz


import numpy as np

from decimal import Decimal

from ETL import remoteLog
from ETL import GetFileName
from ETL import GetLatestParameters
from ETL import SetRun
from ETL import logStep


import click


import os
os.environ['AWS_REGION'] = 'cn-north-1'


import numpy as np

import time





def fit(model, x, batch_size=128, runname=None, sess=None):

    pp = Pool(2)

    tl.layers.initialize_global_variables(sess)

    # In[5]:

    # 32mm*32mm*32mm grid

    i = 0

    if (runname):
        filename, i = GetLatestParameters(runname)

        print "cintinue:", filename
        print "step:", i

        if filename:

            params = load_npz(name=filename)
            assign_params(sess, params, model['net'])

    vl=[]
    kl=[]
    if 'metrics' in model:
        print 'metrics exist'


        metrics=model['metrics']
        kl=list(metrics.iterkeys())
        vl=list(metrics.itervalues())

    while True:
        # resampling

        td = x.next()
        m = td['data']
        label = td['label']

        print m.shape
        print label.shape

        acc = []

        for ii in range(m.shape[0]/batch_size):

            start = time.time()
            [dt, dc] = sess.run([model['top'], model['cost']], feed_dict={
                model['inputs'][0]: m[ii*batch_size:(ii+1)*batch_size],
                model['inputs'][1]: label[ii*batch_size:(ii+1)*batch_size]})
            end = time.time()
            
            dm=sess.run(vl,feed_dict={
                model['inputs'][0]: m[ii*batch_size:(ii+1)*batch_size],
                model['inputs'][1]: label[ii*batch_size:(ii+1)*batch_size]}
                )

            print "acc:", dc, "training time:", ii, end-start
            

            pp.apply_async(logStep, (end-start, dc,kl,dm))

            acc.append(Decimal(np.asscalar(dc)))

        i = i+1
        filename = GetFileName(i)

        print filename

        params = sess.run(model['net'].all_params)

        pp.apply_async(put_to_npz, (params, filename))
        pp.apply_async(remoteLog,(filename, i, {'acc': acc}))


