import numpy as np
import os

if __name__ == '__main__':
    
    # # ----工作目录
    # ls_d = os.listdir('D:/data/银川/rpg')
    # for date in ls_d[:]:
    #     # ----文件列表
    #     path_f = 'D:/data/银川/rain_rate_mlp/'+date+'/'; ls_f = os.listdir(path_f)
    #     # ----save路径
    #     save_path = path_f.replace('rain_rate_mlp', '1h_mlp')
    #     if os.path.exists(save_path) == False:
    #         os.makedirs(save_path)
        
    #     datas = []; dates = []
    #     for fname in ls_f[:]:
    #         qpe = np.load(path_f + fname)
    #         datas.append(qpe); dates.append(fname)
    #         # ----6分钟，加到最后一个时刻
    #         if len(datas) < 10:
    #             continue
    #         elif len(datas) == 10:
    #             temp = np.array(datas)
    #             accu = np.sum(temp * 6/60, axis = 0)
                
    #             np.save(save_path + fname, accu)
    #             datas.pop(0);dates.pop(0) 
    #         # input('go on?')
    
    #%%降雨场次累计
    
    # ----工作目录
    path = 'D:/data/YinChuan/rain_rate_zr/'
    ls_d = os.listdir(path)
    datas = []
    for date in ls_d[:]:
        # ----文件列表
        path_f = path + date + '/'; ls_f = os.listdir(path_f)
        # # ----save路径
        # save_path = path_f.replace('rain_rate_zr', '1h_mlp')
        # if os.path.exists(save_path) == False:
        #     os.makedirs(save_path)
        
        # datas = []; dates = []
        data = []
        for fname in ls_f[:]:
            qpe = np.load(path_f + fname)
            data.append(qpe)
        data = np.array(data)
        sumry = np.sum(data, axis=0)*6/60
        datas.append([date, sumry])
    