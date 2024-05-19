import random
from glob import glob

from tqdm import tqdm

random.seed(1921)

trainset_ro = 0.5
valset_ro = 0.4
testset_ro = 1.0-valset_ro-trainset_ro

rootpath = '/home2/zyl/BraTS19/HGG/npy/*'
output = '../data'
caselist = glob(rootpath)
n_cases = len(caselist)
random.shuffle(caselist)

trainset = caselist[0:int(trainset_ro*n_cases)]
valset = caselist[int(trainset_ro*n_cases):int((trainset_ro+valset_ro)*n_cases)]
testset = caselist[int((trainset_ro+valset_ro)*n_cases):-1]

dataset = {'train':trainset, 'val':valset, 'test':testset}


for phase in tqdm(list(dataset.keys()),position=0,ascii=True,ncols=120):
    caselist = dataset[phase]
    slice_list = []
    fp = open('%s/%s_HGG.txt'%(output,phase),'w')
    for case in tqdm(caselist,position=1,ascii=True,ncols=120):
        case_slice_list = glob(case+'/*')
        case_slice_list.sort()
        for i,piece_path in enumerate(case_slice_list):
            fp.write(piece_path+'\n')
