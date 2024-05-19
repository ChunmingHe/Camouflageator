from glob import glob
import random
import numpy as np
from tqdm import tqdm
random.seed(1921)
trainset_ro = 0.5
valset_ro = 0.4
testset_ro = 1.0-valset_ro-trainset_ro

rootpath = '/home/LiTS/npy/case*'
output = './data/'
caselist = glob(rootpath)
n_cases = len(caselist)
random.shuffle(caselist)

trainset = caselist[0:int(trainset_ro*n_cases)]
valset = caselist[int(trainset_ro*n_cases):int((trainset_ro+valset_ro)*n_cases)]
testset = caselist[int((trainset_ro+valset_ro)*n_cases):-1]

dataset = {'train':trainset, 'val':valset, 'test':testset}


for phase in tqdm(list(dataset.keys()),position=0,ascii=True,ncols=120):
    # with open('./data/%s.txt'%(phase),'r') as fp:
    #     paths = fp.readlines()
    #     casenames = []
    #     for path in paths:
    #         casenames.append((path.split('/'))[-2])
    #     casenames = set(casenames)
    #     paths_tmp = []
    #     for casename in casenames:
    #         paths_tmp.append('/home/LiTS/npy/%s'%(casename))
    #     dataset[phase] = paths_tmp

    caselist = dataset[phase]
    slice_list = []
    for case in tqdm(caselist,position=1,ascii=True,ncols=120):
        case_slice_list = glob(case+'/*')
        for i,piece_path in enumerate(case_slice_list):
            piece = (np.load(piece_path)).item()
            seg_liver = np.where(piece['seg_label']>0,1,0)
            area = np.sum(seg_liver)
            if area > 2500:
                # del[case_slice_list[i]]
                slice_list.append(case_slice_list[i])
    random.shuffle(slice_list)
    with open('%s/%s_for_eval.txt'%(output,phase),'w') as fp:
        for piece in slice_list:
            fp.write(piece+'\n')