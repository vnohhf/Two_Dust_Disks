'''
Author       : vnohhf
Date         : 2021-02-25 22:37
LastEditTime : 2024-05-07 10:46
E-mail       : zry@mail.bnu.edu.cn
Description  : 
Copyright© 2021 vnohhf. ALL RIGHTS RESERVED.
'''

#%% 
from astropy.io import fits 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import pandas as pd
import joblib
import os
from scipy.optimize import curve_fit
import math
import astropy.coordinates as coord
from tqdm import tqdm
import sympy
from astropy import wcs
import time
import astropy.units as u
import healpy as hp
import sys 
sys.path.append("/Users/vnohhf/Documents/Python/package/vnohhf_func")
from vn_data_processing import stable_eliminate
from vn_data_processing import fit_binmid
from vn_binning3D import binning3D
from extinction_coefficient import extinction_coefficient

#%% 导入数据 ======================================================================
grad = joblib.load('MAIN1_grad.pkl.nosync')
grad1D = grad.apply(np.concatenate)
# x和φ需要重新计算
grad1D['x']   = -grad1D.x
grad1D['phi'] = -grad1D['phi']

#* Halo 的消光使用一个常数, 从MAIN4得到
Halo_EBV = 1.52e-3

#%% 设置数据空间范围，用binning3D分bin，保存数据 ========================================
# 定义范围和分bin
Rlim = np.array([5,14])
Zlim = np.array([-2.56,2.56])
offset = 0
galcen_distance = 8.122
temp_array = np.array([sum(0.02*1.2**np.arange(i)) for i in range(19)])
Zdiv_array = np.hstack((-temp_array[::-1],temp_array[1:]))
Rdiv_array = np.arange(Rlim[0], Rlim[1]+0.1, 0.5)
# 开始分bin
bin, plotmatrix, plotmatrix_x, plotmatrix_y, plotmatrix_err = binning3D(grad1D.ebv, grad1D.r, Rdiv_array, grad1D.z, Zdiv_array, binnum_limit=20, method='mean', adaptive_size=True, order=2)
bin['r'], bin['z'] = bin['x'], bin['y']
z_sun = 20.8e-3
#人工剔除的outlier和误差限制
matter  = joblib.load('RZbin.pkl')
outliers = [389,  387, 416,  361, 388, 402,  362]
filters  = np.where(matter.err/matter.value>1)[0]


#%% 单盘/双盘总体拟合 =====================================================================
def SinglePlaneFunc(x,a,z0,h):
    R,Z = x
    # return a * np.exp(-abs(Z-z0)/h) + Halo_EBV
    return a / np.cosh(np.log(1+np.sqrt(2)) * abs(Z-z0)/h)**2 + Halo_EBV

def DoublePlaneFunc(x, a1, h1, a2, h2, z0): 
    R,Z = x
    return a1/np.cosh(np.log(1+np.sqrt(2)) * abs(Z*1e3-z0)/h1)**2 + a2/np.cosh(np.log(1+np.sqrt(2)) * abs(Z*1e3-z0)/h2)**2 + Halo_EBV

# 双盘参数
Double_p,_,_,_,_,_ = joblib.load('disk_fit_result.pkl')

# 单盘参数，和两种结果
Rrange  = np.array([5,6,7,8,9,10,11,12,13,14])
Zarray = (Zdiv_array[1:]+Zdiv_array[:-1])/2
Rarray = (Rdiv_array[1:]+Rdiv_array[:-1])/2
bin['model1'] = np.full([len(Zarray),len(Rarray)], np.nan)
bin['model2'] = np.full([len(Zarray),len(Rarray)], np.nan)
Single_p = np.empty((len(Rrange)-1,3))
for i in range(len(Rrange)-1):
    ind = np.where((Rarray>Rrange[i]) & (Rarray<Rrange[i+1]))[0]
    if i<=4: fitZlim=[-0.3,0.3]
    else:    fitZlim=[-1.5,1.5]
    iLi0 = np.where((~np.isnan(matter.value)) &
                    (matter.r > Rrange[i])    &   (matter.r < Rrange[i+1]) &
                    (matter.z > fitZlim[0])   &   (matter.z < fitZlim[1]))[0]
    iLi = np.setdiff1d(iLi0, np.union1d(filters,outliers))
    Single_p[i],_ = curve_fit(SinglePlaneFunc, np.array([matter.r[iLi],matter.z[iLi]]), np.array(matter.value[iLi]), p0=[0.8,0,0.1], 
                       bounds=[(0,-0.5,0),(np.inf,0.5,1.3)], sigma=matter.err[iLi])
    RZgrid = np.array([[r,z] for z in Zarray for r in Rarray[ind]])
    bin['model1'][:,ind] = SinglePlaneFunc([RZgrid[:,0],RZgrid[:,1]], *Single_p[i]).reshape(bin['model1'][:,ind].shape)
    bin['model2'][:,ind] = DoublePlaneFunc([RZgrid[:,0],RZgrid[:,1]], *Double_p[i]).reshape(bin['model2'][:,ind].shape)

joblib.dump(Single_p,'MAIN3_disk_fit_result.pkl')


#%% 单盘/双盘总体拟合model图和残差 ====================================================================
lower_lim, upper_lim, grad_label, grad_unit = 0.0005, 0.8, 'ΔE(B-V)/Δd', '  [mag/kpc]'
cmap = plt.cm.get_cmap("Spectral_r").copy()
cmap.set_bad('lightgray',0.9)
cmap3 = plt.cm.get_cmap("RdBu").copy()
cmap3.set_bad('lightgray',0.9)
text = {'model1':'Single-disk model', 'model2':'Two-disk model'}

for n,(name,clabel,figprefix,figsuffix,fignum) in enumerate(zip(['model1','model1','model2','model2'],
                                                                ['ΔE(B-V)/Δd','Residuals','ΔE(B-V)/Δd','Residuals'],
                                                                ['Single','Single','Two','Two'],
                                                                ['',' (Residuals)','',' (Residuals)'],
                                                                ['(c)','(d)','(e)','(f)'])):
    fig1, (cax, ax) = plt.subplots(2,1, figsize=(4,4), gridspec_kw={"height_ratios":[0.05,1]}, tight_layout=True, dpi=200)
    #* model图 ------------------------------------------------------------------
    if n in [0,2]:
        im = ax.pcolormesh(*np.meshgrid(Rdiv_array,Zdiv_array), bin[name], cmap=cmap, norm=colors.LogNorm(vmin=lower_lim, vmax=upper_lim))
        cb=plt.colorbar(im,cax=cax,orientation='horizontal',ticklocation='bottom')
        cb.minorticks_on()
        cax.set_title(clabel+grad_unit, fontsize=10)
        # 标注
        ax.text(0.03,0.97, text[name], fontsize=11, c='w', weight='bold', transform=ax.transAxes, va='top', ha='left')
        # 编号标记
        ax.text(0.02,0.03, fignum, fontsize=11, c='k', weight='bold', transform=ax.transAxes, va='bottom', ha='left')

    #* 残差图 -------------------------------------------------------------------------
    if n in [1,3]:
        ori_residuals = (bin['value']-bin[name])
        residuals = ori_residuals
        min_abs_value = -3 #设置一个最小的对数值，绝对值小于这个数的设为此数
        residuals[(0<residuals)&(residuals<10**min_abs_value)] = 10**min_abs_value
        residuals[(0>residuals)&(residuals>-10**min_abs_value)] = -10**min_abs_value
        pos,neg = np.where(ori_residuals>0),np.where(ori_residuals<0)
        residuals[pos] = np.log10(ori_residuals[pos]) - min_abs_value
        residuals[neg] = -(np.log10(abs(ori_residuals[neg]))  - min_abs_value)
        # residuals[(0<residuals)&(residuals<=lower_lim)] = lower_lim #把低于下限的设为下限
        im = ax.pcolormesh(*np.meshgrid(Rdiv_array,Zdiv_array), residuals, cmap=cmap3, vmin=-1.69897, vmax=1.69897)
        
        #* 残差colorbar（在线性刻度下伪装成两个对数刻度） ------------------------------------
        # 创建cb
        cb5=plt.colorbar(im,cax=cax,orientation='horizontal',ticklocation='bottom',extend='both')
        cax.set_title(clabel+grad_unit, fontsize=10)
        # 主刻度
        ticklabelsLi = np.array([0.01,0.05]) #cb需要显示的刻度
        ticksLi = np.log10(ticklabelsLi)-min_abs_value #cb对应的原始刻度
        cb5.set_ticks(np.hstack((-ticksLi[::-1],0,ticksLi))) #主刻度显示
        cb5.set_ticklabels(np.hstack((-ticklabelsLi[::-1],0,ticklabelsLi))) #主刻度labels替换
        # 次刻度
        minorticksLi = np.log10(np.hstack((np.arange(1e-3,11e-3,1e-3),np.arange(1e-2,6e-2,1e-2))))-min_abs_value #创建次刻度对应的原始刻度
        cb5.set_ticks(np.hstack((-minorticksLi[::-1],0,minorticksLi)), minor=True) #次刻度显示

        #* 虚线框和标注 ----------------------------------------------------------------------
        # ax.add_patch(plt.Rectangle((6, -1.2), 7, 0.87, ls='--', lw=1.3, fill=False))
        # ax.add_patch(plt.Rectangle((6, 0.33), 7,  0.9, ls='--', lw=1.3, fill=False))
        ax.plot([6,6,13,13,14,14,12,12,6], [1.2,0.33,0.33,1.2,1.2,2.12,2.12,1.2,1.2], ls='--', lw=1.5, c='k')
        ax.plot([6,6,13,13,14,14,12,12,6], [-1.2,-0.33,-0.33,-1.2,-1.2,-2.12,-2.12,-1.2,-1.2], ls='--', lw=1.5, c='k')
        ax.text(0.03,0.97, text[name], fontsize=11, c='k', weight='bold', transform=ax.transAxes, va='top', ha='left')
        # 编号标记
        ax.text(0.02,0.03, fignum, fontsize=11, c='k', weight='bold', transform=ax.transAxes, va='bottom', ha='left')

    #* setup ------------------------------------------------------------------------------
    ax.plot(galcen_distance,0, c='k', marker=r'$\bigodot$', ms=7, mec='none', zorder =4) # the sun
    ax.set(xlim=Rlim, ylim=Zlim, xlabel= 'R  [kpc]', ylabel='Z  [kpc]')
    ax.minorticks_on()
    fig1.savefig('/Users/vnohhf/Documents/Python/LAMOST_dust_disk/figure/3.1.'+str(n)+'_'+figprefix+'_disk_Model'+figsuffix+'.png')
    fig1.savefig('/Users/vnohhf/Documents/Python/LAMOST_dust_disk/figure/3.1.'+str(n)+'_'+figprefix+'_disk_Model'+figsuffix+'.pdf')

