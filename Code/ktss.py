import math 
import sys 
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.linalg
import scipy.io as sio
from scipy.fftpack import fftn, fftshift, ifftn, ifftshift, fft, ifft
global param

class Matops(np.ndarray): 
    def __new__(cls, *args, **kwargs):
        kwargs.setdefault("dtype", np.complex)
        return super(Matops, cls).__new__(cls, *args, **kwargs)
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'info', None)

#@staticmethod
def ifft2c_mri(x):
    x = fftshift(fft(fftshift(x,0),None, 0),0)/math.sqrt(np.size(x,0))
    x = fftshift(fft(fftshift(x,1),None, 1),1)/math.sqrt(np.size(x,1))
    return x

#@staticmethod
def fft2c_mri(x):
    x = fftshift(ifft(fftshift(x,0),None, 0),0)/math.sqrt(np.size(x,0))
    x = fftshift(ifft(fftshift(x,1),None, 1),1)/math.sqrt(np.size(x,1))
    return x

class Emat_xyz(object):
    #ONE INTERESTING THING: WE ARE EXTENDING A 2-3 DIMENSIONAL ARRAY INTO A 3-4 dIMENSIONAL ARRAY 
    #HOW DO YOU MANAGE THIS  #
    #!TODO experiment with Numpy expand dims
    

    def __init__(self, mask, b1):
        self.mask = np.asarray(mask, order = 'F')
        self.b1 = np.asarray(b1, dtype= complex, order = 'F')
        self.adjoint = 0

    def conj(self): #TODO-1: ufunc for transpose (ct)
        self.adjoint = 1
    
    def __matmul__(self, b): #TODO-2: look up __multiply__ vs __matmul__
        #! Do we need to change whether we do x_array separately... 
        self.x_array = np.zeros(shape = (192, 192, 40, 12), dtype= np.complex128, order = 'F' )
        if self.adjoint:
            result = np.zeros(shape = (192, 192, 40), dtype = np.complex128, order = 'F' )
            for x in range(0, np.size(b,3)):
                self.x_array[:,:,:,x] = np.array(ifft2c_mri(b[:,:,:,x]*(self.mask)), dtype = np.complex128, order = 'F')

            for tt in range(0, np.size(b,2)):
                #res(:,:,tt)=sum(squeeze(x_array(:,:,tt,:)).*conj(a.b1),3)./sum(abs((a.b1)).^2,3) !this is the original fucntion 
                result[:,:,tt] = np.sum(np.multiply(self.x_array[:,:,tt,:].copy().squeeze(),np.conj(self.b1)),axis = 2)/np.sum(abs(self.b1)**2, axis = 2)
        else:  
            result = np.zeros(shape = (192, 192, 40, 12), dtype = np.complex128, order = 'F')
            for tt in range(0, np.size(b,2)):
                for x in range(0, np.size(self.b1,2)): 
                    self.x_array[:,:,tt,x] = b[:,:,tt] *self.b1[:,:,x]

            result = fft2c_mri(self.x_array)
            for x in range(0, np.size(self.b1, 2)):
                result[:,:,:,x] = result[:,:,:,x] * self.mask 
        self.adjoint = 0
        return result


class TempFFT(object):
    def __init__(self, dim):
        self.adjoint = 0
        self.dim = dim


    def conj(self):
        self.adjoint = 1

    def __matmul__(self, b): #TODO-2 linke
        b = np.array(b, order = 'F')
        if type(b) is TempFFT:
            TypeError("Only A can be TempFFT operator")
        if self.adjoint:
            return np.array(ifft(ifftshift(b, axes = self.dim-1), axis = self.dim-1)*math.sqrt(np.size(b, self.dim-1)), order = 'F')
        else:
            return np.array(fftshift(fft(b, axis = self.dim-1), axes = self.dim-1)/math.sqrt(np.size(b, self.dim-1)), order = 'F')
        self.adjoint = 0

    __array_priority__ = 10000

class TVOP(object):
    def __init__(self):
        self.adjoint = 0

    def conj(self):
        self.adjoint = self.adjoint^1

    def adjDx(img):
        res = img[:,[0,-1]]-img
        res[:0] = -img[:,1]
        res[:-1] = img[:, -2]
        return res

    def adjDy(img):
        res = img[[0,-1],:]-img
        res[0:] = -img[1,:]
        res[-1:] = img[-2,:]
        return res

    def adjD(self, img):
        res = np.zeros(np.size(img, 0), np.size(img, 1))
        rets = adjDx(img[:, :, 0]) + adjDy(img[:, :, 1])
        return res
    
    
    def D(self, img):   
        Dx = img[np.r_[1:len(img), -1]] - img
        Dy = img[np.c_[1:len(img), -1]] - img 
        return np.concatenate(Dx, Dy, 2)


    def __matmul__(self, b):
        if self.adjoint:
            res = adjD(b)
        else: 
            res = D(b)
        return res
        self.adjoint = 0

    
    def __multiply__(self, b):
        return self.__matmul__(self, b)
    __array_priority__ = 10000

def start(recon_cs, param):
    x0 = recon_cs.copy('F')
    maxlsiter = 150
    gradToll = 1e-3
    l1Smooth = 1e-15
    param['l1Smooth'] = l1Smooth
    alpha = 0.01
    beta = 0.6
    t0 = 1.0
    k = 0 
    g0 = np.array(grad(x0, param), order = 'F')
    dx = -g0 #there is an issue here because dx is not the right value 
    print("Start Loop")
    while(1):
        f0 = objective(x0, dx, 0, param)
        t = t0  
        f1 = objective(x0, dx, t, param)
        lsiter = 0
        print("second loop")
        value = alpha*t*np.abs(g0.flatten('F').conj()@dx.flatten('F'))
        while (f1 > f0 - value)^2  and (lsiter < maxlsiter): #math.pow((bool(f1 > f0) - alpha*t*np.abs(g0.flatten('F').conj()@dx.flatten('F'))),2)
            lsiter = lsiter + 1
            t = t*beta
            f1 = objective(x0, dx, t, param)
        
        print("exit second loop")
        if lsiter > 2:
            t0 = t0 * beta
        if lsiter < 1:
            t0 = t0/beta

        x0 = x0 + t*dx

        if param["display"] != 0:
            print("ite = {0} cost = {1}  \n".format(str(k), str(f1)))
        
        g1 = np.array(grad(x0, param), order = 'F')
        bk = g1.flatten('F').conj()@g1.flatten('F')/(g0.flatten('F').conj()@g0.flatten('F') + sys.float_info.epsilon)
        g0 = g1
        dx = -g1 + bk*dx
        k = k + 1

        if (k > param["nite"]) or np.linalg.norm(dx.flatten('F')) < gradToll:
            break
    return x0 
    

def objective(x, dx, t, param):
    dxt = t*dx
    xdxt = x + dxt
    step1 = param['E']@xdxt
    w = np.array(step1-param['y'],dtype= np.complex128, order = 'F')
    L2Obj = np.array(w.flatten('F').conj()@w.flatten('F'), order = 'F')
    L1Obj = 0
    if param["L1Weight"] != 0:
        temp = np.array((xdxt), order = 'F')
        w = np.array(param['W']@(temp), order = 'F')
        L1Obj = np.sum(np.add(w.flatten('F').conj()*w.flatten('F'),param['l1Smooth'])**.5)
    else:
        L1Obj = 0
    TVObj = 0
    if param['TVWeight'] != 0:
        w = param['TVWeight']@(x+t*dx)
        TVObj = np.sum(np.add(w.flatten('F').conj()*w.flatten('F'),param['l1Smooth'])**.5)
    else:
        TVObj = 0
    return L2Obj + param['L1Weight']*L1Obj + param['TVWeight']*TVObj

def grad(x, param):
    step1 = np.array(param['E']@x,dtype= np.complex128, order = 'F')
    step2 = np.array(step1 - param['y'], dtype = np.complex128, order = 'F')
    param['E'].conj()
    L2Grad = np.array((param['E']@step2) * 2 , order = 'F')
    L1Grad = 0
    if param["L1Weight"] != 0:
        w = np.array(param['W'] @ x)
        param['W'].conj()
        temp = np.array(w*w.conj())
        L1Grad = np.array(param['W']@w*(np.add(temp,param['l1Smooth'])),dtype= np.complex128, order = 'F')**-0.5
    else:
        L1Grad = 0
   
    TVGrad = 0
    if param['TVWeight'] != 0:
        w = np.array(param['TV'] @ x, order = 'F')
        param['TV'].conj()
        temp = np.array(w*w.conj(), order = 'F')
        TVGrad = np.array(param['TV']@w*(np.add(temp,param['l1Smooth'])),dtype= np.complex128, order = 'F')**-0.5
    else:
        TVGrad = 0

    return np.array(L2Grad + param['L1Weight']*L1Grad + param['TVWeight']*TVGrad, order = 'F')


#!make this into a main 

mat_data = sio.loadmat('data_2d_cardiac_perf.mat')

#mat_dimen = mat_data['kdata'].shape
param = {}
E = Emat_xyz(mat_data['mask'], mat_data['b1'])
param["E"] = E
param["W"] = TempFFT(3)
param["L1Weight"] = 0.05
param["TV"] = TVOP()
param["TVWeight"] = 0
param["y"] = np.array(mat_data['kdata'],dtype= np.complex128, order = 'F')
param["nite"] = 8
param["display"] = 1

E.conj()
param["E"] = E
recon_dft = np.array(param["E"]@param["y"],dtype= complex, order = 'F')
recon_cs = np.array(recon_dft.copy('F'), order = 'F')
for n in range(3):
    recon_cs = start(recon_cs, param) # TODO what is the end data type of recon_cs

recon_cs2 = recon_cs[:,:,0].copy('F')
recon_cs2 = np.concatenate((recon_cs2, recon_cs[:,:,6].copy('F')), 1)
recon_cs2 = np.concatenate((recon_cs2, recon_cs[:,:,12].copy('F')), 1)
recon_cs2 = np.concatenate((recon_cs2, recon_cs[:,:,22].copy('F')), 1)

plt.imshow(np.absolute(recon_cs2), cmap="gray")

plt.savefig('MRi_Test5.png')