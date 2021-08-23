import os
import sys
import cv2
import torch
import random
import numpy as np
import xml.etree.ElementTree as ET

from PIL import  Image
from utils.utils import resize_pad
from torchvision import transforms

class VOCDataset(torch.utils.data.Dataset):
    CLASSES_NAME = (
        "__background__",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )
    def __init__(self,root_dir,resize_size=[800,1333],split='trainval',use_difficult=False,is_train = True, augment = None):
        self.root=root_dir                  #'VOC0712'
        self.use_difficult=use_difficult
        self.imgset=split

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        with open(self._imgsetpath%self.imgset) as f:
            self.img_ids=f.readlines()
        # self.img_ids=[x.strip() for x in self.img_ids]
        self.img_ids = [(self._imgsetpath%self.imgset , x.strip()) for x in self.img_ids]
        self.name2id=dict(zip(VOCDataset.CLASSES_NAME,range(len(VOCDataset.CLASSES_NAME))))
        self.id2name = {v:k for k,v in self.name2id.items()}
        self.resize_size=resize_size
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.train = is_train
        self.augment = augment
        # print("INFO=====>voc dataset init finished  ! !")


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self,index):
        '''
            Return:
                img     : tensor shape(c,h,w)  'rgb'
                boxes   : tensor shape(None,4) (x1,y1,x2,y2)
                classes : tensor shape(None,1)
            Attention:
                PIL to numpy: (w,h,c) -> (h,w,c) 
                         but: rgb -> rgb
        '''
        img_id = self.img_ids[index][1]
        img_path = self._imgpath%img_id
        img = Image.open(img_path)

        anno = ET.parse(self._annopath%img_id).getroot()
        boxes = []
        classes = []
        for obj in anno.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.use_difficult and difficult:
                continue
            _box=obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box=[
                _box.find("xmin").text,
                _box.find("ymin").text,
                _box.find("xmax").text,
                _box.find("ymax").text,
            ]
            TO_REMOVE=1
            box = tuple(
                map(lambda x: x - TO_REMOVE, list(map(float, box)))
            )
            boxes.append(box)

            name=obj.find("name").text.lower().strip()
            classes.append(self.name2id[name])

        boxes=np.array(boxes,dtype=np.float32)
        # import os
        # rank =int(os.environ['RANK'])
        # if rank == 3 and index == 16397:
        #     print('classes', boxes, classes)
        if self.train:
            if self.augment is not None:
                img, boxes = self.augment(img, boxes)
        # import os
        # rank =int(os.environ['RANK'])
        # if rank == 3 and index == 16397:
        #     img.save('3.jpg')
        #     print('img.size', img.size)
        #     print('classes', boxes, classes)
        # print('PIL H: ', img.size[1])  #debug
        img = np.array(img)
        # print('CV2 H:', img.shape[0])  #debug
        img,boxes = resize_pad(img,boxes,self.resize_size)

        img = transforms.ToTensor()(img)
        boxes = torch.from_numpy(boxes)
        classes = torch.LongTensor(classes)
        # return img, boxes, classes
        return img, boxes, classes, img_path, index

    def collate_fn(self, data):
        '''
            Fuction:
                executed while for_loop of dataloader is generated
            Args:
                data:list[(img, boxes, classes),(),(),...]
                     len = batch_size
                     img,boxes,classes from _getitem_
            Returns:
                tuple(img, boxes, classes, img_path)
                (img, boxes, classes, img_path)
            Attention:
                zip(*data): Return tuple
                but we hope img,box,cls transfer to tensor
        '''
        # print("debug...", type(data[0]))
        # print("debug...", *data)
        imgs_list, boxes_list, classes_list, img_path, img_id = zip(*data)
        assert len(imgs_list)==len(boxes_list)==len(classes_list)
        batch_size=len(boxes_list)
        pad_imgs_list=[]
        pad_boxes_list=[]
        pad_classes_list=[]

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        for i in range(batch_size):
            img=imgs_list[i]
            # print(i, img.shape , torch.nn.functional.pad(img,(0,int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.).shape)
            pad_imgs_list.append(transforms.Normalize(self.mean, self.std, inplace=True)(torch.nn.functional.pad(img,(0,int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.)))

        max_num=0
        for i in range(batch_size):
            n=boxes_list[i].shape[0]
            if n>max_num:max_num=n
        for i in range(batch_size):
            pad_boxes_list.append(torch.nn.functional.pad(boxes_list[i],(0,0,0,max_num-boxes_list[i].shape[0]),value=-1))
            pad_classes_list.append(torch.nn.functional.pad(classes_list[i],(0,max_num-classes_list[i].shape[0]),value=-1))
        
        batch_imgs=torch.stack(pad_imgs_list)
        batch_boxes=torch.stack(pad_boxes_list)
        batch_classes=torch.stack(pad_classes_list)

        return batch_imgs,batch_boxes,batch_classes,img_path,img_id

if __name__=="__main__":
    from utils.utils import seeds
    seeds(100)

    dataset = VOCDataset(root_dir='VOC0712', resize_size=[800, 1333],
                               split='train', use_difficult=False, is_train=True, augment=None)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True,
                                               collate_fn=dataset.collate_fn,
                                               num_workers=0)
    for i, data in enumerate(train_loader):
        img, boxes, classes, img_path = data
        print(boxes)
        
    
    #         batch_imgs, batch_boxes, batch_classes = data
    #dataset=VOCDataset("/home/data/voc2007_2012/VOCdevkit/VOC2012",split='trainval')
    # for i in range(100):
    #     img,boxes,classes=dataset[i]
    #     img,boxes,classes=img.numpy().astype(np.uint8),boxes.numpy(),classes.numpy()
    #     img=np.transpose(img,(1,2,0))
    #     print(img.shape)
    #     print(boxes)
    #     print(classes)
    #     for box in boxes:
    #         pt1=(int(box[0]),int(box[1]))
    #         pt2=(int(box[2]),int(box[3]))
    #         img=cv2.rectangle(img,pt1,pt2,[0,255,0],3)
    #     cv2.imshow("test",img)
    #     if cv2.waitKey(0)==27:
    #         break
    #imgs,boxes,classes=eval_dataset.collate_fn([dataset[105],dataset[101],dataset[200]])
    # print(boxes,classes,"\n",imgs.shape,boxes.shape,classes.shape,boxes.dtype,classes.dtype,imgs.dtype)
    # for index,i in enumerate(imgs):
    #     i=i.numpy().astype(np.uint8)
    #     i=np.transpose(i,(1,2,0))
    #     i=cv2.cvtColor(i,cv2.COLOR_RGB2BGR)
    #     print(i.shape,type(i))
    #     cv2.imwrite(str(index)+".jpg",i)