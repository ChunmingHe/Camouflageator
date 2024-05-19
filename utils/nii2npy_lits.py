import os

import numpy as np
import scipy.ndimage as ndimage
import SimpleITK as sitk
from skimage import measure, morphology
from tqdm import tqdm

down_scale = 0.5  # 横断面降采样因子
expand_slice = 20  # 仅使用包含肝脏以及肝脏上下20张切片作为训练样本
slice_thickness = 1  # 将所有数据在z轴的spacing归一化到1mm

def get_boundingbox(mask):
        # mask.shape = [image.shape[0], image.shape[1], classnum]        
        # 删掉小于10像素的目标
        mask_without_small = morphology.remove_small_objects(mask,min_size=10,connectivity=2)
        # mask_without_small = mask
        # 连通域标记
        label_image = measure.label(mask_without_small)
        #统计object个数
        object_num = len(measure.regionprops(label_image))
        boundingbox = list()
        for region in measure.regionprops(label_image):  # 循环得到每一个连通域bbox
            boundingbox.append(region.bbox)
        return object_num, boundingbox

def slice_ct():
    for i in range(131):
        print('processing case {:0>3d}'.format(i))
        path = './raw_data/' 

        Image = sitk.ReadImage(path+'volume-%d'%(i), sitk.sitkInt16)
        ArrayImage = sitk.GetArrayFromImage(Image)

        Label = sitk.ReadImage(path+'segmentation-%d'%(i), sitk.sitkUInt8)
        ArrayLabel = sitk.GetArrayFromImage(Label)

        # 将灰度值在阈值之外的截断
        ArrayImage[ArrayImage > 250] = 250
        ArrayImage[ArrayImage < -200] = -200

        # 对CT数据在横断面上进行降采样,并进行重采样,将所有数据的z轴的spacing调整到1mm
        ArrayImage = ndimage.zoom(ArrayImage, (Image.GetSpacing()[-1] / slice_thickness, 1, 1), order=3)
        ArrayLabel = ndimage.zoom(ArrayLabel, (Label.GetSpacing()[-1] / slice_thickness, 1, 1), order=0)


        #挑出肝脏、肿瘤区域
        label_for_liver = np.where(ArrayLabel==2,1,ArrayLabel)
        label_for_liver = np.where(label_for_liver==1,True,False)
        label_for_tumor = np.where(ArrayLabel==2,True,False)
        
        #肝脏起止点
        liver_range = np.where(label_for_liver==1)
        liver_range = list(set(liver_range[0]))
        liver_range.sort()
        start, end = liver_range[0]-20, liver_range[-1]+20 
        if start < 0:
            start = 0
        if end > ArrayImage.shape[0]:
            end = ArrayImage.shape[0]

        tumor_range = np.where(ArrayLabel==2)
        tumor_range = list(set(tumor_range[0]))
        if tumor_range == []:
            print('volume %d has no tumor area.'%(i))
            has_tumor = False
        else :
            has_tumor = True 
        
        for n in tqdm(range(start,end)):
            slice_dict ={}
            #切片是否含有肝脏
            if n in liver_range:
                n_liver, bbox_liver= get_boundingbox(label_for_liver[n])
                has_liver = True
            else:
                n_liver = 0
                has_liver = False
            
            #切片是否含有肿瘤
            if n in tumor_range:
                n_tumor, bbox_tumor= get_boundingbox(label_for_tumor[n])
                has_tumor = True
            else:
                n_tumor = 0
                has_tumor = False

            slice_dict = {'liver':ArrayImage[n],'seg_label':ArrayLabel[n],
                          'has_tumor':has_tumor,'has_liver':has_liver}            
            for s in range(n_liver):
                dict_name = 'bbox_liver_%d'%(s)
                slice_dict[dict_name] = bbox_liver[s]
            for s in range(n_tumor):
                dict_name = 'bbox_tumor_%d'%(s)
                slice_dict[dict_name] = bbox_tumor[s]

            path = './npy/case{:0>3d}/'.format(i)
            if not os.path.exists(path):
                os.mkdir(path)
            if has_liver:
                if has_tumor:
                    flag = '11'
                else:
                    flag = '10'
            else:
                if has_tumor:
                    flag = '01'
                else:
                    flag = '00'
            filename = '{}{:0>3d}_{}.npy'.format(path,n,flag)
            np.save(filename, slice_dict) 

slice_ct()
