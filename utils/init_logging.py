'''
Description: init a logger
Autor: Au3C2
Date: 2020-08-26 12:16:13
LastEditors: Au3C2
LastEditTime: 2021-10-22 10:32:36
'''

import logging  # 引入logging模块
import os.path

def init_logging(starttime,log_file=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关

    formatter = logging.Formatter("[%(levelname)s]%(asctime)s: %(message)s")
    date = starttime[:10]

    if log_file:
        if not os.path.exists('./log'):
            os.mkdir('./log')
        if not os.path.exists(f'./log/{date}'):
            os.mkdir(f'./log/{date}')
        fh = logging.FileHandler(f'./log/{date}/{starttime}.txt', mode='w')
        fh.setLevel(logging.INFO)  # 输出到file的log等级的开关
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # 输出到console的log等级的开关
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger