'''
Author       : vnohhf
Date         : 2021-02-25 22:37
LastEditTime : 2023-11-24 22:04
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
from scipy.interpolate import interp1d
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
from extinction_coefficient import extinction_coefficient

#* 常数设定
# K_abs: absorption cross section per mass of dust
# (https://www.astro.princeton.edu/~draine/dust/extcurvs/kext_albedo_WD_MW_3.1_60_D03.all)
K_abs = 1.123e4  # unit: cm^2/gram # corresponding lambda = 547 nm
M_sun = 1.9891e33 # unit: gram
μ = 1.37 # the mean molecular weight (Lombardi et al. 2011)
R_step = 0.01
conv_kpc2cm = u.kpc.to(u.cm, 1)

#* 读取数据
Rrange  = np.array([5,6,7,8,9,10,11,12,13,14])
R_array = (Rrange[1:]+Rrange[:-1])/2
mcmc_par,sigma_thin,sigma_thick,sigma_whole,frac_disp,par_disply = joblib.load('disk_fit_result.pkl')
[integral,sl_par,sl_disp] = joblib.load('MAIN2_scale-length parameters.pkl')
exp_func = lambda x,a,b: a*np.exp(-x/b)

#%% 整理HI和H2的径向密度轮廓 =======================================================================
#* HI和H2径向密度轮廓 ------------------------------------------------------------
# 来自H. Nakanishi and Y. Sofue（2016）[自己扣的数据]
# rho: surface density.    h: scale height
HI = pd.read_csv('origin_data/HI surface density vs R (Nakanishi+ 2016, fig4).csv') 
H2 = pd.read_csv('origin_data/H2 surface density vs R (Nakanishi+ 2016, fig4).csv')
HI['h'] = np.array([ 136.0, 124.3, 183.2, 234.7, 226.4, 316.1, 382.1, 505.3, 457.6, 512.2, 570.8, 680.6, 800.5, 856.6, 888., 1039.9, 1196.0, 1344.8, 1485.1, 1525.8, 1589.0, 1700.2, 1669.8, 1662.3, 1707.9, 1778.7, 1794.8, 1911.8, 2009.0, 2092.3, 2111.2]) /2 
HI['h_err'] = np.array([ 52.55, 27.1, 49.5, 34.75, 31.05, 64., 57.3, 82.65, 369.45, 158.05, 202.75, 200.7, 217.45, 224.9, 256.2, 281.15, 392.4, 465.5, 580., 509.85, 564.2, 623.85, 602.8, 615.1, 626.7, 679.2, 743., 878.65, 1064.1, 1335.55, 1431.45]) /2 
H2['h'] = np.array([48,56,68,88,78,102,84,90,186,182,160,222]) /2 
H2['h_err'] = np.array([20,18,14,24,12,10,22,30,20,20,20,27]) /2 
# 把H2.rho用指数函数外推一下
p,_ = curve_fit(exp_func, H2.R[8:], H2.rho[8:])
for i,r in zip(np.arange(12,15),np.arange(12.5, 15, 1)):
    H2.loc[i,['R','rho']] = np.array([r, exp_func(r,*p)])

#来自A. Marasco et al (高斯函数的HWHM)
M17_H2_R = 8.3
M17_H2_HWHM = 52 
M17_H2_HWHM_error = 40 
M17_HI_R = 8.3
M17_HI_HWHM = 217 
M17_HI_HWHM_error = 40


#%% 标高和前人工作比较 =======================================================================
#APOGEE数据结果
par_disply_APO = joblib.load('/Users/vnohhf/Documents/Python/DIB_dr16/MAIN5_disk_fit_result.pkl')[-1]['Dust']

#* 画图 -------------------------------------------------------------------------------------
fig7, ax = plt.subplots(figsize=(6,6.5), facecolor='w', dpi=300)
fig7, ax = plt.subplots(figsize=(5,5), facecolor='w', dpi=300) #ppt格式
#* HI (Thick dust disk) --------------------------------------------------------------------
l1 = ax.errorbar(R_array, par_disply[:,9], yerr = np.vstack((par_disply[:,10],par_disply[:,11])), marker="p", c='C0', ms=7, mec='none', alpha=0.8, ls='none', elinewidth=0.8, zorder=4)
# l2 = ax.errorbar([6.5, 7.5, 8.5, 9.5, 11], par_disply_APO[:,9], yerr = np.vstack((par_disply_APO[:,10],par_disply_APO[:,11])), marker="o", c='w', ms=4, mec='gray', ecolor='gray', ls='none', elinewidth=0.5)
l3 = ax.errorbar(HI.R+0.03, HI.h, yerr=HI.h_err, marker="D", c='w', ms=4, mec='k', mew=1.5, ecolor='k', ls='none', elinewidth=0.5, zorder=3)
l4 = ax.errorbar(M17_HI_R, M17_HI_HWHM, yerr=M17_HI_HWHM_error, marker=".", c='w', mec='k', ecolor='k', ms=9, ls='none', elinewidth=0.5, zorder=2)

#* H2 (Thin dust disk) ---------------------------------------------------------------------
ax.errorbar(R_array, par_disply[:,3], yerr = np.vstack((par_disply[:,4],par_disply[:,5])), marker="p", c='C0', ms=7, mec='none', alpha=0.8, ls='none', elinewidth=0.8, zorder=4)
ax.errorbar(H2.R, H2.h, yerr=H2.h_err, marker="D", c='w', ms=4, mec='k', mew=1.5, ecolor='k', ls='none', elinewidth=0.5, zorder=3)
ax.errorbar(M17_H2_R, M17_H2_HWHM, yerr=M17_H2_HWHM_error, marker=".", c='w', mec='k', ecolor='k', ms=9, ls='none', elinewidth=0.5, zorder=2)
# 分割线
ax.axline([4.5, 110],slope=11, ls='--', lw=1, c='gray')
# 编号标记
ax.text(0.02,0.02, '(a)', fontsize=9, c='k', weight='bold', transform=ax.transAxes, va='bottom', ha='left')

#* setup -----------------------------------------------------------------------------------
ax.text(0.97,0.03, r'Thin dust disk v.s. $\mathbf{H_2}$', fontsize=11, weight='bold', transform=ax.transAxes, va='bottom', ha='right')
ax.text(0.97,0.97, 'Thick dust disk v.s. HI', fontsize=11, weight='bold', transform=ax.transAxes, va='top', ha='right')
ax.set(xlim=(5,12), ylim=(0,500), xlabel='R  [kpc]', ylabel='Scale-height (HWHM)  [pc]')
ax.minorticks_on()
ax.grid(True,ls=':',lw=0.2,zorder=1,color='dimgray')
ax.legend([l1,l3,l4],('This work','Nakanishi & Sofue (2016)','Marasco+ (2017)'), loc=2, fontsize=9)
#
fig7.align_labels()
fig7.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.08)
fig7.savefig('figure/4.1_scale-height comparison.pdf')
fig7.savefig('figure/4.1_scale-height comparison.png')


#%% 计算总质量 =================================================================================
Integral_EBV = np.empty(3)
M_dust = {}
for R1,R2 in [[5,14],[0,30]]:
    for i,(comp,name) in enumerate(zip(['Whole','Thin','Thick'],['尘埃盘总','尘埃薄盘','尘埃厚盘',])):
        for j,coeff in enumerate([sl_par[i]-sl_disp[i,[2,4]],sl_par[i],sl_par[i]+sl_disp[i,[2,5]]]):
            Integral_EBV[j] = 2*np.pi* np.sum( exp_func(np.arange(R1,R2,R_step),*(coeff)) * (np.arange(R1,R2,R_step) * conv_kpc2cm) * (R_step * conv_kpc2cm) ) #最后一个系数：kpc换算到cm
        Integral_Av = 3.1 * Integral_EBV
        M_dust[comp] = np.log(10) / (2.5*K_abs) * Integral_Av / M_sun
        print(str(R1)+'-'+str(R2)+'kpc内, 银河系的'+name+'质量: %.1e (+%.e -%.e) M⊙'%(M_dust[comp][1],*np.diff(M_dust[comp])[::-1]))
    comp = 'Whole'
    if R2==14:
        print(str(R1)+'-'+str(R2)+'kpc内, 银河系的总气尘比: %d (+%d -%d)'%tuple(μ*np.array([5.15e9/M_dust[comp][1], 5.15e9/M_dust[comp][0]-5.15e9/M_dust[comp][1], 5.15e9/M_dust[comp][1]-5.15e9/M_dust[comp][2]])))
    if R2==30:
        print('0-30kpc内, 银河系的总气尘比: %d (+%d -%d)'%tuple(μ*np.array([8e9/M_dust[comp][1], 8e9/M_dust[comp][0]-8e9/M_dust[comp][1], 8e9/M_dust[comp][1]-8e9/M_dust[comp][2]])))
print('M. de Bennassuti et al. (2014) 模拟的银河系尘埃总质量: 1.1e8 M⊙')
print('\n')
print('NakanishiI & Sofue (2016) 估计的银河系 30 kpc 内 HI 质量: 7.2e9 M⊙')
print('NakanishiI & Sofue (2016) 估计的银河系 12 kpc 内 H2 质量: 8.5e8 M⊙')
print('NakanishiI & Sofue (2016) 估计的银河系 30 kpc 内 HI+H2 质量: 8.0e9 M⊙')
print('Bovy & Rix (2013) 估计的银河系气体盘总质量: 7.0e9 M⊙')
print('Draine et al. (2014) 使用尘埃物理模型估算得到的 M31 尘埃盘 (25 kpc内) 总质量: 5.7e7 M⊙')
print('Zhang & Yuan (2020) 估计的 M31 尘埃盘总质量: 6e7 M⊙')


#%% 计算不同R处的质量面密度 ==========================================================================
for j,comp,integral_err, in zip([0,1,2],['Whole','Thin','Thick'],[sigma_whole,sigma_thin,sigma_thick]):
    x,y,yerr = R_array, integral[comp], integral[comp] * (integral_err[:,0]+integral_err[:,1])/2

# 开始计算
Mass = pd.DataFrame(index=range(len(Rrange[:-1])), columns=['Whole','Thin','Thick','HI','H2','H']).astype(np.float64) #dust：拟合值。H：文献数据
emp_Mass = pd.DataFrame(index=range(len(Rrange[:-1])), columns=['Whole','Whole_err1','Whole_err2','Thin','Thin_err1','Thin_err2','Thick','Thick_err1','Thick_err2']).astype(np.float64) #dust：实测点
for i in range(len(Rrange[:-1])):
    R1,R2,R_step = Rrange[i],Rrange[i+1],0.01
    S = np.pi*(R2**2-R1**2) * 1e3**2
    #* 尘埃的质量面密度
    for j,(comp,integral_err) in enumerate(zip(['Whole','Thin','Thick'],
                                               [sigma_whole,sigma_thin,sigma_thick])):
        # 拟合值
        Integral_EBV = 2*np.pi* np.sum( exp_func(np.arange(R1,R2,R_step),*sl_par[j]) * (np.arange(R1,R2,R_step) * conv_kpc2cm) * (R_step * conv_kpc2cm) ) #最后一个系数：kpc换算到cm
        Integral_Av = 3.1 * Integral_EBV
        Mass[comp].loc[i] = np.log(10) / (2.5*K_abs) * Integral_Av / M_sun / S
        # 实测点
        for integral_value, suffix in zip(np.append(integral[comp][i],integral[comp][i] * integral_err[i]), ['','_err1','_err2']):
            Integral_EBV = integral_value * np.pi* (R2**2-R1**2) * conv_kpc2cm**2 #最后一个系数：kpc换算到cm
            Integral_Av = 3.1 * Integral_EBV
            emp_Mass[comp+suffix].loc[i] = np.log(10) / (2.5*K_abs) * Integral_Av / M_sun / S
    #* H2、HI的质量面密度
    Mass['H2'].iloc[i] = interp1d(H2.R, H2.rho)([(R1+R2)/2])[0]
    Mass['HI'].iloc[i] = interp1d(HI.R, HI.rho)([(R1+R2)/2])[0]
Mass['H'] = Mass['H2'] + Mass['HI']

#* 面密度随R变化图 ----------------------------------------------------------------------------
fig, ax = plt.subplots(1,1, figsize=(5,4), facecolor='w', dpi=300)
for i,(comp,complabel,ls,c) in enumerate(zip(['H','HI','H2','Whole','Thin','Thick'],
                                           ['HI+H$_{2}$','HI','H$_{2}$','Dust (Whole)','Dust (Thin)','Dust (Thick)'],
                                           ['--','--','--','-','-','-'],
                                           ['C0','C2','C1','C0','C2','C1'])):
    ax.plot(R_array, Mass[comp], label=complabel, ls=ls, c=c, lw=1)
    if i >= 3:
        ax.errorbar(R_array, emp_Mass[comp], yerr=emp_Mass[[comp+'_err1',comp+'_err2']].to_numpy().T, ms=8, alpha=0.9, elinewidth=0.7, color=c, fmt='.', mec='none', zorder=1)
ax.set_yscale('log')
ax.set(ylim=[5e-3,200], ylabel=r'Mass surface density  [${\rm M_⊙/pc^2}$]')
ax.legend(loc=2,ncol=2,fontsize=8)
ax.grid(True,ls=':',lw=0.2,zorder=1,color='dimgray',alpha=0.7)
ax.set(xlabel='R  [kpc]')
ax.minorticks_on()
fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
fig.savefig('figure/4.2_surface density comparison.pdf')
fig.savefig('figure/4.2_surface density comparison.png')



#%% 气尘质量比(GDR)随R变化图 ======================================================================
def expfunc1(x,a,b):
    return 10**(a*x+b)

fig, ax = plt.subplots(1,1, figsize=(5,4), facecolor='w', dpi=300)
line = {}
for i,(comp1,comp2,c) in enumerate(zip(['HI','H','H2'],['Thick','Whole','Thin'],['C2','C0','C1'])):
    GDR_abs_err = np.empty((len(Rrange[:-1]),2))
    y = μ*Mass[comp1]/emp_Mass[comp2]
    # 散点
    for j in range(len(Rrange[:-1])):
        rho_with_err = emp_Mass[comp2][j] + [-1,1] * emp_Mass[[comp2+'_err1',comp2+'_err2']].to_numpy()[j] #加上误差的rho
        GDR_abs_err[j] = abs( μ*Mass[comp1][j] / rho_with_err[::-1] - y[j] )
        if (i==0) & (j >= 7): mfc='w' # 标记出不使用的点
        else: mfc=c
        ax.errorbar(R_array[j], y[j], yerr=GDR_abs_err[j].reshape(2,1), ms=8, alpha=0.9, elinewidth=0.8, color=c, fmt='.', mfc=mfc, zorder=1)
    if i==0:
        gdr_p,_ = curve_fit(expfunc1, R_array[:-2], y[:-2], sigma=np.mean(GDR_abs_err, axis=1)[:-2], maxfev = 10000)
    else:
        gdr_p,_ = curve_fit(expfunc1, R_array, y, sigma=np.mean(GDR_abs_err, axis=1), maxfev = 10000)
    xdata = np.linspace(5,15,30)
    print(expfunc1(5,*gdr_p))
    label = comp2+' dust disk: $log(γ) = %.2fR + %.2f$'%tuple(gdr_p)
    ax.plot(xdata, expfunc1(xdata,*gdr_p), ms=8, alpha=0.9, color=c, mec='none', zorder=1, label=label)
# 文献值1 (Giannetti+ (2017))
Giannetti = {}
Giannetti['R'] = np.array([6,15])
Giannetti['GDR'] = 10**(0.087*Giannetti['R']+1.44)
Giannetti['GDR_with_err'] = np.vstack((10**((0.087-0.007)*Giannetti['R']+1.44-0.2), Giannetti['GDR'], 10**((0.087+0.007)*Giannetti['R']+1.44+0.2)))
ax.errorbar(Giannetti['R'], Giannetti['GDR'], yerr=np.diff(Giannetti['GDR_with_err'], axis=0), label='Giannetti+ (2017)', ms=4, alpha=0.9, elinewidth=0.7, color='k', marker='s', lw=0, mec='none', zorder=1) # 2017A&A...606L..12G
# 编号标记
ax.text(0.02,0.03, '(c)', fontsize=11, c='k', weight='bold', transform=ax.transAxes, va='bottom', ha='left')
# setup
ax.legend(fontsize=10)
ax.set(ylim=[1,0.7e5],ylabel='Gas-to-dust mass ratio (γ)')
ax.minorticks_on()
ax.set_yscale('log')
ax.grid(True,ls=':',lw=0.2,zorder=1,color='dimgray')
ax.set(xlabel='R  [kpc]', xticks=(np.arange(5,16)))
fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.12)
fig.savefig('figure/4.3_GTD ratio comparison.pdf')
fig.savefig('figure/4.3_GTD ratio comparison.png')


#%% 整理气体和尘埃的径向密度轮廓到一个DF -> pf =======================================================
# 这样的话，pf可以完全替代前面的HI，H2，Mass。但是由于工程麻烦，就不改了。之后优先使用pf
#* gas ---------------------------------------------------------------------------------------
col_Li=[]
for i,comp in enumerate(['H','HI','H2']):
    for var in ['R','rho','h','h_err']:
        col_Li.append([comp,var])
for i,comp in enumerate(['Whole','Thin','Thick']):
    for var in ['R','rho','rho_err1','rho_err2','h','h_err1','h_err2']:
        col_Li.append([comp,var])
pf = pd.DataFrame(columns=pd.MultiIndex.from_arrays(np.array(col_Li).T)).astype(np.float64) # R_profile
pf['HI'] = HI
pf['H2'] = H2
# 计算'H'的参数，即HI+H2
pf.loc[:,[['H','R']]] = pf['HI']['R']
pf.loc[:,[['H','rho']]] = list(np.nansum([pf['HI']['rho'], pf['H2']['rho']], axis=0))

#* dust --------------------------------------------------------------------------------------
for i,comp in enumerate(['Whole','Thin','Thick']):
    pf.loc[:len(R_array)-1,[[comp,'R']]] = R_array
# 导入h
par_index = [3,4,5,9,10,11]
i=0
for comp in ['Thin','Thick']:
    for var in ['h','h_err1','h_err2']:
        pf.loc[:len(R_array)-1,[[comp,var]]] = par_disply[:,par_index[i]]
        i+=1
# 导入rho
i=0
for comp in ['Whole','Thin','Thick']:
    for var in ['rho','rho_err1','rho_err2']:
        pf.loc[:len(R_array)-1,[[comp,var]]] = emp_Mass[emp_Mass.keys()[i]]
        i+=1

# 假设H2和薄盘标高完全一样，HI和厚盘也完全一样
# pf.loc[:len(R_array)-1,[['Thick','h']]] = interp1d(pf['HI']['R'], pf['HI']['h'])(R_array)
# pf.loc[:len(R_array)-1,[['Thin','h']]] = interp1d(pf['H2']['R'], pf['H2']['h'])(R_array)


#%% vd(R)图: 体密度随R变化图=======================================================================
Zrange  = np.linspace(0,2,11)
Z_array = (Zrange[1:]+Zrange[:-1])/2
interest_R = [6.5,8.5,9.5,10.5,11.5]
vd = pd.DataFrame(index=interest_R, columns=['H2','HI','Thin','Thick','H','Whole']) #volume density

# 已知在Z方向，气体密度分布符合 rho = ∫ (a / np.cosh(np.log(1+np.sqrt(2)) * abs(Z*1e3)/h)**2) dZ
# 已知h，但是a在文中没有给出，现通过拟合函数求解a
# 气体符合的密度轮廓函数（z方向）：
def z_profile(z,a,h):
    return a / np.cosh(np.log(1+np.sqrt(2)) * abs(z)/h)**2

fig, axes = plt.subplots(3,2, figsize=(8,12), facecolor='w', tight_layout=True, dpi=300)
for axi,comp,zmax in zip(axes.flat,['H2','Thin','HI','Thick','H','Whole'],[400,400,1500,1500,1e3,1e3]):
    for R in interest_R:
        # 选出非nan的行
        useI = np.where(~np.isnan(pf[comp].R))[0]
        pf_use = pf.loc[useI,comp]
        # 得到该 R 时的 h,rho
        h = interp1d(pf_use.R, pf_use.h)([R])[0]
        rho = interp1d(pf_use.R, pf_use.rho)([R])[0]
        # 解算中盘面密度 a0
        z_step = 1
        zLi = np.arange(0,1500,z_step)
        a0 = rho / np.sum( 2 / np.cosh(np.log(1+np.sqrt(2)) * abs(zLi)/h)**2 * z_step )
        # 求解 vd(R)
        if comp in ['H2','HI']:
            vd.loc[R,comp] = μ * z_profile(zLi,a0,h)
        elif comp in ['Thin','Thick']:
            vd.loc[R,comp] = z_profile(zLi,a0,h)
        elif comp in ['H']:
            vd.loc[R,comp] = vd.loc[R,'H2'] + vd.loc[R,'HI']
        elif comp in ['Whole']:
            vd.loc[R,comp] = vd.loc[R,'Thin'] + vd.loc[R,'Thick']
        axi.plot(zLi, vd.loc[R,comp], label='R = '+str(R))
    # setup
    axi.legend(loc=0, title=comp)
    axi.set(xlim=[0,zmax], xlabel='Z  [pc]', ylabel='Volume density  [M$_⊙/pc^3$]')
    axi.minorticks_on()
    axi.grid(True,ls=':',lw=0.2,zorder=1,color='dimgray')
fig.savefig('figure/4.4_Variation of volume density with R.png')


#%% 气尘质量比(GDR)随Z变化图 ======================================================================
fig, ax = plt.subplots(1,3, figsize=(12,4), facecolor='w', tight_layout=True, dpi=300)
for i,(comp1,comp2,zmax) in enumerate(zip(['H2','HI','H',],['Thin','Thick','Whole',],[400,1500,1e3])):
    for R in interest_R:
        ax[i].plot(zLi, vd.loc[R,comp1]/vd.loc[R,comp2], label='R = '+str(R))
        # setup
        if i==0:
            ax[i].set(ylim=[0,100])
        ax[i].legend(loc=0, title=comp1+'/'+comp2)
        ax[i].set(xlim=[0,zmax], xlabel='Z  [pc]', ylabel='Gas-to-Dust Ratio')
        ax[i].minorticks_on()
        ax[i].grid(True,ls=':',lw=0.2,zorder=1,color='dimgray')
fig.savefig('figure/4.5_Variation of GDR with Z.pdf')
fig.savefig('figure/4.5_Variation of GDR with Z.png')

#%%
