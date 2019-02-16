# coding: utf8
import numpy as np 
from scipy import ndimage


import boto3


ddb=boto3.resource('dynamodb')
table=ddb.Table('vfx_seg')
from boto3.dynamodb.conditions import Key, Attr

res=table.scan(FilterExpression=Attr('train_include').exists())
data=res['Items']


while 'LastEvaluatedKey' in res:
    res = table.scan(FilterExpression=Attr('train_include').exists(),ExclusiveStartKey=res['LastEvaluatedKey'])
    data.extend(res['Items'])


Samples=[(d['image'],d['label']) for d in data]

from random import shuffle

def train_input():
    import imageio


    while True:
        shuffle(Samples)


        print len(Samples)
        iml=[]
        labels=[]
        
        for i in range(64):

            img,label=Samples[i]
            print img
            print label
            
            dimg=imageio.imread(img)
            
            dlabel=imageio.imread(label)

            print dlabel.shape
            print dimg.shape
         
            
            iml.append(dimg.reshape(1,1920,1080,3))
            labels.append(dlabel.reshape(1,1920,1080))

        tx=np.concatenate(iml,axis=0)
        ty=np.concatenate(labels,axis=0)  

        print tx.shape
        print ty.shape

        affine=np.array([1,1079/511,1919/511.0,1])
        affine2=np.array([1,1079/511,1919/511.0])
    
        tsx=ndimage.affine_transform(tx[0:64],affine,output_shape=(64,512,512,3))
        tsy=ndimage.affine_transform(ty[0:64],affine2,output_shape=(64,512,512))
   
        tsy=tsy.reshape(-1,512,512,1)

        
        

        

        tx=(tsx-128.0)/128.0
        ty=tsy/255.0

        tx=tx.astype(np.float32)
        ty=ty.astype(np.float32)
        
        yield {'data':tx,
        'label':ty}



if __name__=="__main__":
    g=train_input()

    for i in range(10):
        d=g.next()
        print i
        print d['data'].shape,d['label'].shape
