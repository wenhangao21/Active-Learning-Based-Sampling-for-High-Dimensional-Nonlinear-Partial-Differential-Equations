import scipy.io as sio
import numpy

for k in range(30):
    mat_contents = sio.loadmat('l2seq'+ str(k)+'.mat')
    array = mat_contents[str(k)+'l2seq']
    sio.savemat('l2seq'+ str(k)+'.mat', mdict={'erl2seq'+ str(k): array})


for k in range(30):
    mat_contents1 = sio.loadmat('timeseq'+ str(k)+'.mat')
    array1 = mat_contents1[str(k)+'timeseq']
    sio.savemat('timeseq'+ str(k)+'.mat', mdict={'ertimeseq'+ str(k): array1})