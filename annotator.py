
import cv2
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from core import init_model,predict
import os, glob

#Forbidden  Key: QSFKL
class Annotator(object):
    def __init__(self ,input_dir, model,if_sis=False,if_cuda=True,output_dir=None):

        self.model,self.if_sis,self.if_cuda,self.output_dir=model,if_sis,if_cuda,output_dir
        self.imgs_path = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
        self.imgs_path += sorted(glob.glob(os.path.join(input_dir, "*.png")))
        self.file = Path(input_dir).name
        self.imgs = ([np.array(Image.open(img_path)), img_path] for img_path in self.imgs_path)  # generator
        self.img, self.img_path = next(self.imgs)
        self.clicks = np.empty([0,3],dtype=np.int64)
        self.pred = np.zeros(self.img.shape[:2],dtype=np.uint8)
        self.merge =  self.__gene_merge(self.pred, self.img, self.clicks)

        os.makedirs(self.output_dir, exist_ok=True)  # creat directory if not exist

    def __gene_merge(self,pred,img,clicks,r=9,cb=2,b=2,if_first=True):
        pred_mask=cv2.merge([pred*255,pred*255,np.zeros_like(pred)])
        result= np.uint8(np.clip(img*0.7+pred_mask*0.3,0,255))
        if b>0:
            contours,_=cv2.findContours(pred,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result,contours,-1,(255,255,255),b)
        for pt in clicks:
            cv2.circle(result,tuple(pt[:2]),r,(255,0,0) if pt[2]==1 else (0,0,255),-1)
            cv2.circle(result,tuple(pt[:2]),r,(255,255,255),cb) 
        if if_first and len(clicks)!=0:
            cv2.circle(result,tuple(clicks[0,:2]),r,(0,255,0),cb) 
        return result

    def __update(self):
        self.ax1.imshow(self.merge)
        self.fig.canvas.draw()

    def __reset(self):
        self.clicks =  np.empty([0,3],dtype=np.int64)
        self.pred = np.zeros(self.img.shape[:2],dtype=np.uint8)
        self.merge =  self.__gene_merge(self.pred, self.img, self.clicks)
        self.__update()

    def __predict(self):
        self.pred = predict(self.model,self.img,self.clicks,if_sis=self.if_sis,if_cuda=self.if_cuda)
        self.merge =  self.__gene_merge(self.pred, self.img, self.clicks)
        self.__update()

    def __on_key_press(self,event):
        if event.key=='ctrl+z':
            self.clicks=self.clicks[:-1,:]
            if len(self.clicks)!=0:
                self.__predict()
            else:
                self.__reset()
        elif event.key=='ctrl+r':
            self.__reset()
        elif event.key=='escape':
            plt.close()
        elif event.key=='enter':
            if self.output_dir is not None:
                save_path = os.path.join(self.output_dir, os.path.basename(self.img_path))
                Image.fromarray(self.pred*255).save(save_path)
                print('save mask in [{}]!'.format(save_path))
            try:
                self.img, self.img_path = next(self.imgs)
                self.__reset()
            except:
                plt.close()

    def __on_button_press(self,event):
        if (event.xdata is None) or (event.ydata is None):return
        if event.button==1 or  event.button==3:
            x,y= int(event.xdata+0.5), int(event.ydata+0.5)
            self.clicks=np.append(self.clicks,np.array([[x,y,(3-event.button)/2]],dtype=np.int64),axis=0)
            self.__predict()

    def main(self):
        self.fig = plt.figure('Annotator')
        self.fig.canvas.mpl_connect('key_press_event', self.__on_key_press)
        self.fig.canvas.mpl_connect("button_press_event",  self.__on_button_press)
        self.fig.suptitle('( file : {} )'.format(self.file),fontsize=16)
        self.ax1 = self.fig.add_subplot(1,1,1)
        self.ax1.axis('off')
        self.ax1.imshow(self.merge)
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Annotator for FCANet")
    parser.add_argument('--input_dir', type=str, default='test.jpg', help='input image directory')
    parser.add_argument('--output_dir', type=str, default='test_mask.png', help='output mask directory')
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'res2net'], help='backbone name (default: resnet)')
    parser.add_argument('--sis', action='store_true', default=False, help='use sis')
    parser.add_argument('--cpu', action='store_true', default=False, help='use cpu (not recommended)')
    args = parser.parse_args()

    model = init_model('fcanet',args.backbone,'./pretrained_model/fcanet-{}.pth'.format(args.backbone),if_cuda=not args.cpu)
    anno=Annotator(input_dir=args.input_dir ,model=model, if_sis=args.sis, if_cuda=not args.cpu,output_dir=args.output_dir)
    anno.main()
