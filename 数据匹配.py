import numpy as np
import os
import datetime as dt
import pandas as pd

if __name__ == "__main__":
    
    # # ----工作目录
    # ls_d = os.listdir('D:/data/银川/rpg')
    # for date in ls_d[:1]:
    #     # ----标准时间戳: utc+8
    #     ls_t = []; year = int(date[:4]); month = int(date[4:6]); day = int(date[6:])
    #     for i in range(24):
    #         ls_t.append(dt.datetime(year, month, day, i, 0, 0))
    #     ls_t.append(ls_t[-1]+dt.timedelta(hours=1))
        
    #     # ----文件列表
    #     path_f = 'D:/data/银川/1h_zr/'+date+'/'; ls_f = os.listdir(path_f)
        
    #     temp = []
    #     for fname in ls_f[:]:
    #         qpe = np.load(path_f + fname)
    #         tsR = fname.split('_')[4]
    #         tsR = dt.datetime.strptime(tsR, '%Y%m%d%H%M%S') + dt.timedelta(hours = 8) # utc 转 utc+8
            
    #         for tsG in ls_t:
    #             if tsG > tsR and tsG-tsR <= dt.timedelta(minutes = 6):
    #                 temp.append([tsG, fname])
    #     # temp = np.array(temp)
    #%%    
    
    # ----站点及其坐标
    st_infos = pd.read_excel('D:/data/YinChuan/银川/银川2016-2022降雨数据/20230218清华大学数据/站点经纬信息.xlsx')
    df = pd.DataFrame(columns = ['stnm', 'lon', 'lat', 'timestamp', 'hourly rainfall', '1h_zr', '1h_mlp'])
    
    for i in range(len(st_infos)):
        num = i
    # num = 3
        st = st_infos['站号'][num]
        lon = st_infos['经度'][num];# lon1 = int(lon/10000); lon2 = int((lon - lon1*10000)/100); lon3 = lon%100
        # lon = lon1 + lon2/60 + lon3/60/60
        lat = st_infos['纬度'][num];# lat1 = int(lat/10000); lat2 = int((lat - lat1*10000)/100); lat3 = lat%100
        # lat = lat1 + lat2/60 + lat3/60/60
        # df.loc[0,:3] = [st, lon, lat]
        if lon > 105.8 and lon < 106.6 and lat > 38.2 and lat < 38.9:
            
            
            # ----该站点所有的雨量计观测
            path_rain = 'D:/data/YinChuan/gauges/'; ls_rain = os.listdir(path_rain)
            for f_rain in ls_rain[:]:
                df_rain = pd.read_excel(path_rain + f_rain)
                temp = df_rain.loc[ df_rain['台站号'] == st , (['时间', '过去1小时降水量'])]; temp.columns = ['timestamp', 'hourly rainfall']
                df = pd.concat([df, temp])
            df.iloc[:, 0] = st; df.iloc[:, 1] = lon; df.iloc[:, 2] = lat
            df['timestamp'] = df['timestamp'].astype(str)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # df.to_excel(str(st)+'_dataset.xlsx')
            
            #%%
            
            # ----雷达的时间戳列表
            path_date = 'D:/data/YinChuan/1h_zr/'; ls_date = os.listdir(path_date)
            ts = []
            for date in ls_date:
                path_radar = path_date + date + '/'; ls_radar = os.listdir(path_radar)
                
                for fname in ls_radar:
                    tsR = fname.split('_')[4]
                    # tsR = dt.datetime.strptime(tsR, '%Y%m%d%H%M%S') + dt.timedelta(hours = 8) # utc 转 utc+8
                    tsR = dt.datetime.strptime(tsR, '%Y%m%d%H%M%S') # GMT世界时
                    ts.append([tsR, path_radar + fname])
            ts = pd.DataFrame(ts, columns=['timestamp', 'fpath'])
            
            
            #%%
            
            
            for i in range(len(df)):
                    
                tsG = df.iloc[i, 3]
                d = tsG - ts.loc[ tsG > ts['timestamp'], ['timestamp']]
                d = pd.concat([d, ts.loc[ tsG <= ts['timestamp'], ['timestamp']] - tsG])
                loc = d.idxmin()
                
                fpath = ts.iloc[loc, 1].values[0]
                qpezr = np.load(fpath)
                iy = 100+int((lat - 38.4797)/0.005); ix = 100+int((lon - 106.2147)/0.005)
                df.loc[df['timestamp'] == tsG, ['1h_zr']] = np.mean(qpezr[iy-1:iy+2,ix-1:ix+2])
            
                fpath = fpath.replace('zr', 'mlp')
                qpezr = np.load(fpath)
                iy = 100+int((lat - 38.4797)/0.005); ix = 100+int((lon - 106.2147)/0.005)
                df.loc[df['timestamp'] == tsG, ['1h_mlp']] = np.mean(qpezr[iy-1:iy+2,ix-1:ix+2])
            df.to_excel('D:/data/YinChuan/1h_analysis/'+str(st)+'_dataset.xlsx')
