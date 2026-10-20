import cv2, numpy as np, torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
class Preprocessor:
    """Lab 1 cell 72: prepare uint8 RGB, augment, finalize and CHW tensor."""
    def __init__(self,size):
        self.size=size
        self.mean=np.array([.485,.456,.406],np.float32)
        self.std=np.array([.229,.224,.225],np.float32)
    def prepare(self,bgr):
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2]
        interp=cv2.INTER_AREA if h>self.size or w>self.size else cv2.INTER_LINEAR
        return cv2.resize(rgb,(self.size,self.size),interpolation=interp)
    def finalize(self,rgb):return (rgb.astype(np.float32)/255.-self.mean)/self.std

def load_rgb_sample(row):
    im=cv2.imread(row['path'])
    if im is None:raise ValueError('Unreadable image: '+row['path'])
    if row.get('crop') is not None:
        x1,y1,x2,y2=row['crop'];h,w=im.shape[:2];dx=(x2-x1)*.05;dy=(y2-y1)*.05
        left,top=max(0,int((x1-dx)*w)),max(0,int((y1-dy)*h))
        right,bottom=min(w,max(left+1,int((x2+dx)*w))),min(h,max(top+1,int((y2+dy)*h)))
        im=im[top:bottom,left:right].copy()
    return im

class FrameDataset(torch.utils.data.Dataset):
    """Lab 1 cell 104 adapted to frame and human-box crop tasks."""
    def __init__(self,samples,size,train=False):
        self.samples=samples;self.pre=Preprocessor(size)
        self.tf=transforms.Compose([transforms.RandomHorizontalFlip(),
                  transforms.ColorJitter(brightness=.15,contrast=.15)]) if train else None
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        row=self.samples[i];img=self.pre.prepare(load_rgb_sample(row))
        if self.tf:img=np.array(self.tf(Image.fromarray(img)))
        x=torch.from_numpy(np.ascontiguousarray(self.pre.finalize(img))).permute(2,0,1).float()
        return x,int(row['label']),i