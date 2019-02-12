import numpy as np 
from scipy import ndimage

def train_input():
    fl=np.load('vfx_seg.npz')
    tx=fl['tx']
    ty=fl['ty']

    print tx.shape
    print ty.shape

    affine=np.array([1,1079/511,1919/511.0,1])
    affine2=np.array([1,1079/511,1919/511.0])
    
    tsx=ndimage.affine_transform(tx[0:64],affine,output_shape=(64,512,512,3))
    tsy=ndimage.affine_transform(ty[0:64],affine2,output_shape=(64,512,512))
   
    tsy=tsy.reshape(-1,512,512,1)


    del tx
    del ty
    
    print tsx.shape
    print tsy.shape



    while True:

        
        

        

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