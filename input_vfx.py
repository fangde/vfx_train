import numpy as np 

def train_input():
    fl=np.load('../vfx_seg.npz')
    tx=fl['tx']
    ty=fl['ty']

    print tx.shape
    print ty.shape

    
    
    tsx=tx[0:128]
    tsy=ty[0:128].reshape(-1,1080,1920,1)


    del tx
    del ty
    
    print tsx.shape
    print tsy.shape



    while True:

        px=np.random.randint(1080-512,size=1)
        py=np.random.randint(1920-512,size=1)
        px=px[0]
        py=py[0]

        print px,py
        
        tdx=tsx[:,px:px+512,py:py+512,:]
        tdy=tsy[:,px:px+512,py:py+512,:]

        print tsx.shape,tsy.shape

        tx=(tdx-128.0)/128.0
        ty=(tdy/255.0)

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
