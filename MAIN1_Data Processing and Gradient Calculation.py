'''
Author       : vnohhf
Date         : 2021-02-07 21:32
LastEditTime : 2024-05-06 14:44
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
from extinction_coeffcient import extinction_coeffcient

#%% 导入数据 =================================================================
#* LAMOST -------------------------------------------------------------------
LG = pd.DataFrame()
with fits.open('/Users/vnohhf/Documents/AstroDataBase.nosync/LAMOST dr7/egaia.dr7.edr3_revisedXdist.fits') as F1:
    LG['obsid']         = F1[1].data.field('obsid').astype(np.int64)
    LG['source_id']     = F1[1].data.field('source_id').astype(np.int64)
    LG['l']             = F1[1].data.field('GL').astype(np.float64)
    LG['b']             = F1[1].data.field('GB').astype(np.float64)
    LG['snr']           = F1[1].data.field('snrg').astype(np.float64)
    LG['teff']          = F1[1].data.field('teff').astype(np.float64)
    # LG['para']          = F1[1].data.field('parallax').astype(np.float64) + 26e-3
    # LG['parae']         = F1[1].data.field('parallax_error').astype(np.float64)
    LG['G']             = F1[1].data.field('PHOT_G_MEAN_MAG_CORRECTED').astype(np.float64)
    LG['d']             = F1[1].data.field('rpgeo').astype(np.float64)/1e3
    LG['d_b']           = F1[1].data.field('b_rpgeo_dis').astype(np.float64)/1e3
    LG['d_B']           = F1[1].data.field('B_rpgeo_disa').astype(np.float64)/1e3
    LG['sfdebv']        = F1[1].data.field('ebv').astype(np.float64)

#* APOGEE -------------------------------------------------------------------
APO = pd.DataFrame()
with fits.open('/Users/vnohhf/Documents/AstroDataBase.nosync/APOGEE dr17/allStar-dr17-synspec_rev1.fits') as F2:
    APO['id']     = F2[1].data.field('APOGEE_ID')
    APO['field']  = F2[1].data.field('FIELD')
    APO['snr']    = F2[1].data.field('SNR').astype(np.float64)
    APO['l']      = F2[1].data.field('GLON').astype(np.float64)
    APO['b']      = F2[1].data.field('GLAT').astype(np.float64)
    APO['ra']     = F2[1].data.field('RA').astype(np.float64)
    APO['dec']    = F2[1].data.field('DEC').astype(np.float64)
    APO['sfdebv'] = F2[1].data.field('SFD_EBV').astype(np.float64)
    # APO['d']      = F2[1].data.field('GAIAEDR3_R_MED_GEO').astype(np.float64)*0.001
    # APO['d_b']    = F2[1].data.field('GAIAEDR3_R_LO_GEO').astype(np.float64)*0.001
    # APO['d_B']    = F2[1].data.field('GAIAEDR3_R_HI_GEO').astype(np.float64)*0.001
with fits.open('/Users/vnohhf/Documents/AstroDataBase.nosync/APOGEE dr16/apogee_astroNN-DR17.fits') as F3:
    APO['d']  = F3[1].data.field('weighted_dist').astype(np.float64)*0.001
    APO['de'] = F3[1].data.field('weighted_dist_error').astype(np.float64)*0.001
APO['EBVsp'] = joblib.load('/Users/vnohhf/Documents/Python/DIB_dr17/MAIN1_APO_extinciton.pkl')[:,2] /1.14
APO['d'][APO['d']<0] = np.nan

#* APOGEE DR16 -------------------------------------------------------------------
# [indexuse,APO16] = joblib.load('/Users/vnohhf/Documents/Python/DIB_dr16/MAIN0_APO_catalog.pkl') 
# APO16[['ebprp','EBVsp']] = joblib.load('/Users/vnohhf/Documents/Python/DIB_dr16/MIAN1_APO_extinciton.pkl') 

# com = pd.merge(APO, APO16['id'], on='id', how="inner")
# com.drop_duplicates(subset=['id'],keep=False)


#%% 匹配 Extinction_law 中得到的 E(BP-RP) ===================================================================================
sys.path.append('/Users/vnohhf/Documents/Python/Extinction_coefficient')
from createDataFrame import createCAT
CAT, _, useLi, _, _, EColor, _, _ = createCAT('CAT')
CAT['EBPRPsp'] = EColor['BP-G']+ EColor['G-RP']
# 星表合并
common = pd.merge(LG,CAT[['obsid','EBPRPsp']],on='obsid',how='left')
common['EBVsp'] = common.EBPRPsp / extinction_coeffcient('BP-RP', Teff=common.teff, EBV=common.sfdebv)
LG['EBVsp'] = common['EBVsp']/1.14

del common,CAT,EColor,_

#%% 样本选择 =====================================================================================================
#LAMOST
usei1 = np.where((LG.snr > 10) & (LG.EBVsp > -0.5) &
                 ((LG.d-LG.d_b)/LG.d < 0.3) & ((LG.d_B - LG.d)/LG.d<0.3) #距离精度好于30% (损失1.3%)
                )[0]
#APOGEE
usei2 = np.where((APO.snr > 40) & (APO.EBVsp > -0.5) & (APO.de/APO.d < 0.3)
                 )[0] #距离精度好于30%，损失3.8%
#LAMOST,APOGEE,Gaia数据整合
# usei3 = np.setdiff1d(usei2, np.where(((350<APO.l) | (APO.l<20)) & (abs(APO.b)<15))[0])
LAG = pd.concat([LG.iloc[usei1],APO.iloc[usei2]], ignore_index=True) 

#%% 比较高银纬的sfd_ebv和APOGEE_ebv、LAMOST_ebv ====================================================================
import sys 
import random
sys.path.append("/Users/vnohhf/Documents/Python/package/vnohhf_func")
from vn_data_processing import fit_binmid
fig2,ax = plt.subplots(1,3,figsize=(13,4),tight_layout=True)
import matplotlib.cm as cmap
# *EBV_SFD vs EBV_gaia ------------------------------------------------------------------------------------------
for i,data,label,ls,theind in zip([0,1], [LG,APO], ['LAMOST','APOGEE'], ['-','--'], [usei1,usei2]):
    high_b_ind = theind[abs(data.b.iloc[theind])>60] #选择区域中的星 
    Z = abs(data.d.iloc[high_b_ind]*np.sin(data.b.iloc[high_b_ind]/180*np.pi))
    far_list = high_b_ind[Z>1.2]  #银盘距Z>1.2
    #画散点
    if i==0: 
        random.seed(1) 
        far_list_rand = far_list[random.sample(range(len(far_list)), round(len(far_list)/20))]
    else:
        far_list_rand = far_list
    xdata = data.sfdebv/1.14
    ydata = data.EBVsp
    ax[i].scatter(xdata[far_list_rand],ydata[far_list_rand],marker='.',c='k',edgecolors='none',s=11,alpha=0.4,zorder=1)
    ax[i].plot(np.array([0,0.14]),np.array([0,0.14]),ls='--',c='C0',alpha=0.8,zorder=2,label='1:1 reference')  #1:1参考线
    bin_x, bin_y, p, pcov, retainedLi = fit_binmid(xdata[far_list],ydata[far_list], np.arange(0.004,0.05,0.004),upsigma=2,lowsigma=2,mode='proportional')
    ax[i].scatter(bin_x, bin_y, marker='.',c='w',edgecolors='C0',alpha=0.9, s=130, lw=2 ,zorder=5, label='Median values')
    ax[i].set_xlim(0,0.06)
    ax[i].set_ylim(-0.02,0.08)
    ax[i].tick_params(labelsize=12) 
    ax[i].set_xlabel(r'$E(B-V)_{\rm SFD}$  [mag]', fontsize=16)
    ax[i].set_ylabel(r'$E(B-V)_{\rm '+label+'}$  [mag]', fontsize=16)
    ax[i].minorticks_on()
    #ax[i].grid(True,ls='-',lw=0.2,zorder=1,color='dimgray')
    ax[i].legend(loc = 2, fontsize=12)

# *EBV差值直方图 ----------------------------------------------------------------------------------
    def gaussian(x, depth, lamb, sigma): #高斯函数
        return depth*np.exp(-(x-lamb)**2/(2*sigma**2))
    # 直方图
    xdata = data.sfdebv.iloc[far_list] - data.EBVsp.iloc[far_list]
    histdata = ax[2].hist(xdata,histtype='step',weights = np.ones_like(xdata)/len(xdata), ls='-', color='C'+str(i),alpha=0.7, bins=np.arange(-0.08,0.08,0.004))
    # 高斯函数    
    binx = (histdata[1][1:] + histdata[1][:-1])/2
    p,_ = curve_fit(gaussian, binx, histdata[0], p0=[0.1, 1e-3, 0.015])
    ax[2].plot(np.linspace(-0.08,0.08,120),gaussian(np.linspace(-0.08,0.08,120),*p),linewidth=1,alpha=0.9,ls='--',label='SFD - '+label)
    # steup
    ax[2].set_xlim(-0.08,0.08)
    ax[2].set_xlabel('$\Delta E(B-V)$  [mag]', fontsize=16)
    ax[2].set_ylabel('Proportion', fontsize=16)
    if label=='LAMOST':
        plt.text(0.05,0.95, 'SFD$-$'+label+'\nμ = %.3f\nσ = %.3f'%tuple([p[1],abs(p[2])]), c='C'+str(i), fontsize=12, ha='left', va='top', transform=ax[2].transAxes, linespacing = 1.4)
    if label=='APOGEE':
        plt.text(0.95,0.95, 'SFD$-$'+label+'\nμ = %.3f\nσ = %.3f'%tuple([p[1],abs(p[2])]), c='C'+str(i), fontsize=12, ha='right', va='top', transform=ax[2].transAxes, linespacing = 1.4)
        
    ax[2].grid(True,ls='-',lw=0.2,zorder=1,color='dimgray')
    ax[2].minorticks_on()
    ax[2].tick_params(labelsize=12)
#
fig2.savefig('/Users/vnohhf/Documents/Python/LAMOST_dust_disk/figure/1.2_Compare EBV in high b.pdf')
fig2.savefig('/Users/vnohhf/Documents/Python/LAMOST_dust_disk/figure/1.2_Compare EBV in high b.png')


#%% 用healpix划分视线方向，求消光梯度 [16min] ========================================
nside = 64
LAG['hpi'] = hp.ang2pix(nside, np.radians(90-LAG.b).to_numpy(), 
                               np.radians(LAG.l).to_numpy())
sight_list = [np.array(v) for v in LAG.groupby('hpi').groups.values()]
sightline_Num = len(sight_list)

#* 按照距离分bin，获得中值 --------------------------------------------------------
d_interval = np.hstack((np.arange(0,1,0.1), 1.1**np.arange(0,25)))+0.5 #对数划分2 (更细，用于XY图)
d_interval = np.hstack((np.arange(0,0.75,0.15), 0.75*1.2**np.arange(22))) #对数划分(2023版)
d_interval = np.hstack((np.arange(0,1,0.1), 1.1**np.arange(0,29))) #对数划分1 (更细，用于XY图)
di = pd.DataFrame()
di_list = [[[] for _ in range(len(d_interval)-2)] for _ in range(sightline_Num)]
for key in ['d','l','b','ebv','wei']:
    di[key] = [np.nan * np.empty((len(d_interval)-2)) for _ in range(sightline_Num)]

for i,li in enumerate(tqdm(sight_list)):
    for j in range(len(d_interval)-3):
        di_list[i][j] = li[(LAG.d[li] > d_interval[j]) & (LAG.d[li] < d_interval[j+1])]
        if len(di_list[i][j]) < 5: #如果数目不够，延长一段距离范围
            di_list[i][j] = li[(LAG.d[li] > d_interval[j]) & (LAG.d[li] < d_interval[j+2])]
        if len(di_list[i][j]) < 5: #如果数目不够，延长一段距离范围
            di_list[i][j] = li[(LAG.d[li] > d_interval[j]) & (LAG.d[li] < d_interval[j+3])]
        if len(di_list[i][j]) == 0: continue
        #di开头变量是一个视场中每一个di点包含的信息  # @用的是中值
        di.wei[i][j] = len(di_list[i][j])
        di.d  [i][j] = np.median(LAG.d    [di_list[i][j]])
        di.l  [i][j] = np.median(LAG.l    [di_list[i][j]])
        di.b  [i][j] = np.median(LAG.b    [di_list[i][j]])
        di.ebv[i][j] = np.median(LAG.EBVsp[di_list[i][j]])
        temp_l = LAG.l[di_list[i][j]]
        if (np.max(temp_l) > 350) and (np.min(temp_l) < 10): #当field跨过ra=360°的情况,ra全部变成360附近的数
            temp_l[temp_l < 30] += 360
            di.l[i][j] = np.median(temp_l)

#* 按照个数分bin，获得中值 -------------------------------------------------------
# di = pd.DataFrame()
# for key in ['d','l','b','ebv','ebv_err']:
#     di[key] = [np.array([]) for _ in range(sightline_Num)]

# for i,li in enumerate(tqdm(sight_list)):
#     index_list = np.argsort(LAG.d[li].to_numpy())
#     di_num = len(index_list)//5
#     index_mat = index_list[:di_num*5].reshape(di_num,5)
#     for j,di_list in enumerate(li[index_mat]):
#         #di开头变量是一个视场中每一个di点包含的信息  # @用的是中值
#         di.d      [i] = np.append(di.d      [i], np.nanmedian(LAG.d  [di_list]))
#         di.b      [i] = np.append(di.b      [i], np.nanmedian(LAG.b  [di_list]))
#         di.ebv    [i] = np.append(di.ebv    [i], np.nanmedian(LAG.EBVsp[di_list]))
#         di.ebv_err[i] = np.append(di.ebv_err[i], np.nanstd(LAG.EBVsp[di_list]))
#         #当field跨过l=360°的情况,l全部变成360附近的数
#         temp_l = LAG.l[di_list]
#         if (np.max(temp_l) > 350) and (np.min(temp_l) < 10): 
#             temp_l[temp_l < 10] += 360
#         di.l[i] = np.append(di.l[i], np.median(temp_l))
        

#%% 求梯度 ======================================================================
grad = pd.DataFrame([[[] for _ in range(11)] for _ in range(sightline_Num)],
                    columns=['d','ebv','ebve','l','b','r','wei','x','y','z','phi'])
for i,li in enumerate(tqdm(sight_list)):
    di_use = np.where((di.wei[i] >= 5) & ~np.isnan(di.ebv[i]))[0]  # @需要原始源数di.wei大于等于5
    for k in range(len(di_use)-1):
        kk = [di_use[k], di_use[k+1]]
        if np.diff(di.d[i][kk]) != 0:
            grad.d[i]  .append( np.mean(di.d  [i][kk]) )
            grad.l[i]  .append( np.mean(di.l  [i][kk])%360) #考虑field跨过ra = 360°的情况
            grad.b[i]  .append( np.mean(di.b  [i][kk]) )
            grad.wei[i].append( np.sum (di.wei[i][kk]) )
            grad.ebv[i].append((np.diff(di.ebv[i][kk]) / np.diff(di.d[i][kk]))[0])
    # 坐标系转换 l,b -> R,Z
    tempcoord = coord.Galactic(l = grad.l[i]  *u.degree,
                               b = grad.b[i]  *u.degree,
                               distance = grad.d[i] *u.kpc).transform_to(coord.Galactocentric)
    grad.r  [i] = np.sqrt(tempcoord.x.value**2 + tempcoord.y.value**2)
    grad.x  [i] = tempcoord.x.value
    grad.y  [i] = tempcoord.y.value
    grad.z  [i] = tempcoord.z.value
    grad.phi[i] = np.arctan(grad.y[i]/grad.x[i])
    
joblib.dump(grad,'MAIN1_grad.pkl.nosync')


#%% 2个方向的示例 =================================================================
fig,axes = plt.subplots(1,2, figsize=(10,4),dpi=250)
for i,ax,ylim in zip([14786,5135],axes.flat,[[0,1.2],[0,0.1]]):
    # di_show = np.where(di.wei[i]>=5)[0]
    di_show = np.where(~np.isnan(di.ebv[i]))[0]
    ax.scatter(LAG.d[sight_list[i]], LAG.EBVsp[sight_list[i]], s=6, c='darkgray', alpha=0.7, edgecolor='none', label="Stars")
    ax.scatter(di.d[i], di.ebv[i], marker='D', s=15, c='C0', alpha=0.2, edgecolor='none')
    ax.scatter(di.d[i][di_show], di.ebv[i][di_show], marker='D', s=18, c='C0', alpha=0.7, edgecolor='none', label="Medians in bins")
    #text
    # ax.text(0.95,0.95, '$(l,b)=(%.1f^\circ,%.1f^\circ)$ '%tuple([np.nanmedian(di.l[i]),np.nanmedian(di.b[i])]), va='top', ha='right', transform = ax.transAxes)
    # setup
    ax.minorticks_on()
    ax.grid(True,ls=':',lw=0.2,zorder=1,color='dimgray')
    ax.axhline(0,lw=0.3,alpha=0.7,zorder=1,color='dimgray')
    ax.set_xlim([0,5]) 
    ax.set_ylim(ylim)
    ax.set_xlabel('Distance  [kpc]') 
    ax.set_ylabel(r'$E(B-V)$  [mag]') 
    ax.legend(title='$(l,b)=(%.1f^\circ,%.1f^\circ)$ '%tuple([np.nanmedian(di.l[i]),np.nanmedian(di.b[i])]), loc=1, fontsize=8)
plt.subplots_adjust(wspace=0.2,left=0.1,right=0.95,bottom=0.15,top=0.95)
fig.savefig('/Users/vnohhf/Documents/Python/LAMOST_dust_disk/figure/1.3_sightline example.pdf')
fig.savefig('/Users/vnohhf/Documents/Python/LAMOST_dust_disk/figure/1.3_sightline example.png')


#%% 在lb空间观察grad，看看是不是有颗粒度不合适的问题 ====================================
i1,i2 = 12341,7020
#* 变量为距离画图
fig, ax = plt.subplots(1,3, figsize=(10,4),tight_layout=True, dpi=250)
grad1D = grad.apply(np.concatenate)
im = ax[0].scatter(grad1D.l, grad1D.b, s=3, edgecolor='none', c=grad1D.d, vmin=0, vmax=5, cmap='Spectral_r')
cb = plt.colorbar(im, label='d [kpc]')
ax[0].set_xlim(150,160)
ax[0].set_ylim(-33,-23)
ax[0].set_xlabel('l [deg]')
ax[0].set_ylabel('b [deg]')
#
for i,axi in zip([i1,i2], ax[1:]):
    axi.scatter(grad1D.l, grad1D.b, s=8, edgecolor='none', c=grad1D.d, vmin=0, vmax=5, cmap='Spectral_r')
    axi.set_xlim(np.nanmedian(di.l[i])+[-0.5,0.5])
    axi.set_ylim(np.nanmedian(di.b[i])+[-0.5,0.5])
    axi.set_xlabel('l [deg]')
#
fig.savefig('/Users/vnohhf/Documents/Python/LAMOST_dust_disk/figure/1.4_在lb空间观察grad.png')

#* 变量为梯度画全天图
import matplotlib
fig, ax = plt.subplots(1,1, figsize=(6,4),tight_layout=True, dpi=250)
grad1D = grad.apply(np.concatenate)
pos = np.where(grad1D.ebv>0)[0]
im = ax.scatter(grad1D.l[pos], grad1D.b[pos], s=3, edgecolor='none', c=grad1D.ebv[pos], norm = matplotlib.colors.LogNorm(vmin=5e-4, vmax=0.7), cmap='Spectral_r')
cb = plt.colorbar(im, label='d [kpc]')
ax.set_xlabel('l [deg]')
ax.set_ylabel('b [deg]')
fig.savefig('/Users/vnohhf/Documents/Python/LAMOST_dust_disk/figure/1.5_在lb空间观察grad(全天).png')


#%% 
fig, ax = plt.subplots(1,1, figsize=(8,4), tight_layout=True, dpi=250, subplot_kw={'projection':'mollweide'})
LG.l[LG.l>180] = LG.l[LG.l>180] - 360
APO.l[APO.l>180] = APO.l[APO.l>180] - 360
ax.scatter(np.radians(LG.l)[usei1], np.radians(LG.b)[usei1], s=0.05, ec='none', rasterized=True)
ax.scatter(np.radians(APO.l)[usei2], np.radians(APO.b)[usei2], s=0.05, ec='none', rasterized=True)
ax.set(xlabel='l  [deg]',ylabel='b  [deg]')
ax.grid(True,ls='-',lw=0.2,zorder=1,color='dimgray')
fig.savefig('/Users/vnohhf/Documents/Python/LAMOST_dust_disk/figure/1.6_spatial distribution of stars.pdf')
fig.savefig('/Users/vnohhf/Documents/Python/LAMOST_dust_disk/figure/1.6_spatial distribution of stars.png')
