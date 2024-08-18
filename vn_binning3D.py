'''
Author       : vnohhf
Date         : 2022-11-17 14:46
LastEditTime : 2023-09-26 16:29
E-mail       : zry@mail.bnu.edu.cn
Description  : Copyright© vnohhf. ALL RIGHTS RESERVED.
'''

# 导入
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit  
import sys 
sys.path.append("/Users/vnohhf/Documents/Python/package/vnohhf_func")
from vn_data_processing import stable_eliminate

#%% 将数据在XYZ三维空间分bin的函数,包含自适应扩大bin ============================================
# 输入XYZ方向的分割值(Z方向默认不分割)；输出完成分bin的矩阵
def binning3D(Value, XLi, XdivLi, YLi, YdivLi, ZLi=[], ZdivLi=[-np.inf,np.inf], 
              method='median', #取bin数值的方法：平均或者中位数
              gaussian=False, #是否对bin内点做高斯拟合
              binnum_limit=20, #每个bin包含的最小数目
              offset=0, #bin.value的常数补偿
              adaptive_size=False, #是否使用自适应扩大bin尺寸的算法
              order=2, #自适应扩大bin尺寸至2倍或者3倍原大小
              outputplot=True, #是否输出绘图矩阵
              fit_direction='z', #是否使用高斯拟合的维度是哪一个
              ): 
    
    #高斯函数
    def gaussian_fun(x,a,b,c):
        return a*np.exp(-(x-b)**2/(2*c**2))

    # 初始化
    if len(ZLi)==0:
        ZLi = np.ones_like(XLi)
    bin = dict()
    if method=='mean':   func = np.nanmean
    if method=='median': func = np.nanmedian
    bin['value'], bin['err'], bin['x'], bin['y'], bin['z'] = [np.full([len(ZdivLi)-1,len(YdivLi)-1,len(XdivLi)-1,], np.nan) for _ in range(5)]
    if gaussian:
        bin['A'], bin['lamb'], bin['sigma'], bin['value'], bin['A_err'], bin['lamb_err'], bin['sigma_err'], bin['ew_err'], = [np.full([len(ZdivLi)-1,len(YdivLi)-1,len(XdivLi)-1,], np.nan) for _ in range(8)]

    #* 分bin统计的子函数 --------------------------------------------------------------------------
    Xindex, Yindex, Zindex = [np.digitize(Li, divLi)-1 for Li,divLi in zip([XLi, YLi, ZLi],[XdivLi, YdivLi, ZdivLi])]
    indLi = [[z,y,x] for z in range(len(ZdivLi)-1) for y in range(len(YdivLi)-1) for x in range(len(XdivLi)-1)]
    def cal_bin(expendLi):
        for z,y,x in indLi:
            if np.isnan(bin['value'][z,y,x]):  # 如果无数据
                for x2,y2 in np.array([x,y])+expendLi:
                    if (x2 < len(XdivLi)) & (y2 < len(YdivLi)) & np.isnan(bin['value'][z, y:y2, x:x2]).all():
                        insidebin = np.where((Xindex>=x) & (Xindex<=(x2-1)) & (Yindex>=y) & (Yindex<=(y2-1)) & (Zindex==z))[0]
                        in3sigma = insidebin[stable_eliminate(Value[insidebin])] #insidebin中剔除3σ外的
                        # 修改bin中数据
                        if (len(in3sigma) >= binnum_limit):
                            bin['x'][z,y:y2,x:x2] = func(XLi[in3sigma])
                            bin['y'][z,y:y2,x:x2] = func(YLi[in3sigma])
                            bin['z'][z,y:y2,x:x2] = func(ZLi[in3sigma])
                            if not gaussian:
                                bin['value'][z,y:y2,x:x2] = func(Value[in3sigma])
                                bin['err'][z,y:y2,x:x2] = np.nanstd(Value[in3sigma]) / np.sqrt(len(in3sigma))
                            elif gaussian:
                                if fit_direction in ['x','X']: array = XLi[in3sigma]
                                if fit_direction in ['y','Y']: array = YLi[in3sigma]
                                if fit_direction in ['z','Z']: array = ZLi[in3sigma]
                                try:
                                    p, pcov = curve_fit(gaussian_fun, array, Value[in3sigma], maxfev=8000, 
                                                        p0=[np.nanmax(Value[in3sigma]),0,np.nanstd(Value[in3sigma])],
                                                        bounds=[[0,-0.5,0],[np.inf,0.5,np.inf]])
                                except RuntimeError:
                                    continue
                                else: 
                                    bin['A']    [z,y:y2,x:x2] = p[0]
                                    bin['lamb'] [z,y:y2,x:x2] = p[1]
                                    bin['sigma'][z,y:y2,x:x2] = p[2]
                                    bin['value'][z,y:y2,x:x2] = abs(np.sqrt(2*np.pi) * p[0] * p[2]) #value即为等效面积EW
                                    perr = np.sqrt(np.diag(pcov))
                                    bin['A_err']    [z,y:y2,x:x2] = perr[0]
                                    bin['lamb_err'] [z,y:y2,x:x2] = perr[1]
                                    bin['sigma_err'][z,y:y2,x:x2] = perr[2]
                                    bin['ew_err'][z,y:y2,x:x2] = np.sqrt(2*np.pi) * bin['value'][z,y:y2,x:x2] * np.sqrt((perr[0]/p[0])**2+(perr[2]/p[2])**2) #相对误差
        return bin
    
    # 不使用自适应扩大bin尺寸的算法，计算结果
    expendLi =  np.array([[1,1]])
    bin = cal_bin(expendLi)
    # 对于低于binnum_limit的bin，使用自适应扩大bin尺寸的算法，牺牲空间分辨率尽量保证有数据
    if adaptive_size == True:
        if order==2: expendLi = np.array([[2,1],[1,2],[2,2]]) #优先将x方向往后延长一格，再y方向延长一格,自适应扩大bin尺寸至2倍原大小
        if order==3: expendLi = np.array([[2,1],[1,2],[2,2],[3,2],[2,3],[3,3]]) #优先将x方向往后延长一格，再y方向延长一格,自适应扩大bin尺寸最多至3倍原大小
        bin = cal_bin(expendLi)
    
    #* 构造用来画热图的矩阵 plotmatrix 和 plotmatrix_err ---------------------------------
    if outputplot:
        #热图分辨率 
        reso_x = 100 #pixel/kpc
        reso_y = 1000 #pixel/kpc
        #把YdivLi整理成整数数列
        show_Xdiv = np.array(np.around(XdivLi-min(XdivLi),decimals=2)*reso_x ,dtype=int)
        show_Ydiv = np.array(np.around(YdivLi-min(YdivLi),decimals=3)*reso_y ,dtype=int)
        plotmatrix, plotmatrix_err = dict(), dict()
        for z in range(len(ZdivLi)-1):
            plotmatrix[z], plotmatrix_err[z] = np.full((2, max(show_Ydiv), max(show_Xdiv)), np.nan).astype(np.float64)
            for y in range(len(YdivLi)-1):
                for x in range(len(XdivLi)-1):
                    if method!='gaussian':
                        plotmatrix    [z][show_Ydiv[y]:show_Ydiv[y+1],show_Xdiv[x]:show_Xdiv[x+1]] = bin['value'][z,y,x] + offset
                        plotmatrix_err[z][show_Ydiv[y]:show_Ydiv[y+1],show_Xdiv[x]:show_Xdiv[x+1]] = bin['err'][z,y,x]
                    elif method=='gaussian':
                        plotmatrix    [z][show_Ydiv[y]:show_Ydiv[y+1],show_Xdiv[x]:show_Xdiv[x+1]] = bin['value'][z,y,x] + offset
                        plotmatrix_err[z][show_Ydiv[y]:show_Ydiv[y+1],show_Xdiv[x]:show_Xdiv[x+1]] = bin['err'][z,y,x]                    
        plotmatrix_x = np.linspace(XdivLi[0],XdivLi[-1],max(show_Xdiv))
        plotmatrix_y = np.linspace(YdivLi[0],YdivLi[-1],max(show_Ydiv))
    
    #如果Z方向不分bin（即为预设值），取消输出值的字典形式
    if len(ZdivLi)==2: 
        for key in bin.keys():
            bin[key] = bin[key][0]
        if outputplot: 
            plotmatrix, plotmatrix_err = plotmatrix[0], plotmatrix_err[0]
    # 输出
    if outputplot: 
        return bin, plotmatrix, plotmatrix_x, plotmatrix_y, plotmatrix_err
    else: 
        return bin
