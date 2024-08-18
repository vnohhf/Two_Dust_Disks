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
from scipy.interpolate import interp1d
from scipy import interpolate 
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

#* 考虑 Warp observed by Chen+2019 [放弃使用] --------------------------------------
def z_w(R,phi): 
    if R<9: #使用幂律
        R_w = 7.72
        φ_w = 17.5 /180*np.pi #注意：角度制弧度制
        a,b = 0.060, 1.33
        z_w = a*(R - R_w)**b * np.sin(phi - φ_w)
    elif R>=9: #使用线性
        R_w = 9.26
        φ_w = 17.4 /180*np.pi 
        a,b = 0.148, 1
        z_w = a*(R - R_w)**b * np.sin(phi - φ_w)
    return z_w

grad1D['z_w'] = np.full_like(grad1D['z'], np.nan)
for i in range(len(grad1D.z)):
    grad1D['z_w'][i] = z_w(grad1D.r[i], grad1D.phi[i]) 
grad1D['z_w'][np.isnan(grad1D['z_w'])] = 0
# grad1D['z'] = grad1D['z'] + grad1D['z_w']

#* 考虑中平面位移 -----------------------------------------------------------------
# miniXLi,miniYLi,smooth_Lamb = joblib.load('MAIN6_平滑后的中平面位移.pkl')

# nonan = np.where(~np.isnan(smooth_Lamb.flatten()))[0]
# grad1D['z_offset'] = interpolate.griddata((miniXLi.flatten()[nonan], miniYLi.flatten()[nonan]), 
#                       smooth_Lamb.flatten()[nonan], np.array([grad1D.x,grad1D.y]).T, method='linear')
# grad1D['z'] = grad1D['z'] - grad1D['z_offset']

#%% 设置数据空间范围，用binning3D分bin，保存数据 ========================================
# 定义范围和分bin
Rlim = np.array([5,14])
Zlim = np.array([-2.56,2.56])
offset = 0
galcen_distance = 8.122
temp_array = np.array([sum(0.02*1.2**np.arange(i)) for i in range(19)])
temp_array = np.array([sum(0.015*1.15**np.arange(i)) for i in range(24)])
Zdiv_array = np.hstack((-temp_array[::-1],temp_array[1:]))
Rdiv_array = np.arange(Rlim[0], Rlim[1]+0.1, 0.5)
# 开始分bin
bin, plotmatrix, plotmatrix_x, plotmatrix_y, plotmatrix_err = binning3D(grad1D.ebv, grad1D.r, Rdiv_array, grad1D.z, Zdiv_array, binnum_limit=20, method='mean', adaptive_size=True, order=2)
bin['r'], bin['z'] = bin['x'], bin['y']
plotmatrix[plotmatrix_err/plotmatrix > 3] = np.nan #去除误差过大的
z_sun = 20.8e-3
# 保存
joblib.dump(pd.DataFrame({'value':bin['value'].flatten(),'err':bin['err'].flatten(),'r':bin['r'].flatten(),'z': bin['z'].flatten()}),'RZbin.pkl')


#%% RZ分布图=========================================================================
lower_lim, upper_lim, grad_label, grad_unit = 0.0005, 0.8, 'ΔE(B-V)/Δd', '  [mag/kpc]'
for n,(clabel,figname,fignum) in enumerate(zip([grad_label,grad_label,'Error  '],['scatter','heatmap','errormap'],['(a)','(b)',''])):
    fig1, (cax, ax) = plt.subplots(2,1, figsize=(4,4), gridspec_kw={"height_ratios":[0.05,1]}, tight_layout=True, dpi=300)
    cmap = plt.cm.get_cmap("Spectral_r").copy()
    cmap.set_bad('lightgray',0.9)
    cmap2 = plt.cm.get_cmap("magma_r").copy()
    cmap2.set_bad('lightgray',0.9)
    cmap3 = plt.cm.get_cmap("RdBu").copy()
    cmap3.set_bad('lightgray',0.9)

    #* 散点图 ---------------------------------------------------------------------------
    if n==0:
        pos = np.random.permutation(np.where((grad1D.ebv)>0)[0])
        neg = np.where((grad1D.ebv)<0)[0]
        im = ax.scatter(grad1D.r[pos], grad1D.z[pos], c=grad1D.ebv[pos], s=2, alpha=0.7, cmap=cmap, edgecolors='none', zorder=3, norm=colors.LogNorm(vmin=lower_lim, vmax=upper_lim), rasterized=True)
        ax.scatter(grad1D.r[neg], grad1D.z[neg], s=2, alpha=0.2, c='gray', edgecolors='none', zorder=2, rasterized=True) #负数设为灰色

    #* 热图与误差图 -----------------------------------------------------------------------
    # 把两个矩阵低于下限的设为下限
    plotmatrix[(0<plotmatrix)&(plotmatrix<=lower_lim)] = lower_lim 
    plotmatrix_err[(0<plotmatrix_err)&(plotmatrix_err<=lower_lim)] = lower_lim 
    #把加上offset之后的负数设为一个很小的值，这样在热图显示为灰色而不是白色
    plotmatrix[plotmatrix<0] = lower_lim/100 
    if n==1:
        im = ax.pcolormesh(*np.meshgrid(plotmatrix_x, plotmatrix_y), plotmatrix, cmap=cmap, norm=colors.LogNorm(vmin=lower_lim, vmax=upper_lim))
    if n==2:
        im = ax.pcolormesh(*np.meshgrid(plotmatrix_x, plotmatrix_y), plotmatrix_err, cmap=cmap2, norm=colors.LogNorm(vmin=lower_lim, vmax=upper_lim))

    #* setup ----------------------------------------------------------------------------
    ax.plot(galcen_distance,0, c='k', marker=r'$\bigodot$', ms=8, mec='none', zorder =4) # the sun
    ax.set(xlim=Rlim, ylim=Zlim, xlabel= 'R  [kpc]', ylabel='Z  [kpc]')
    ax.minorticks_on()
    ax.grid(True,ls=':',lw=0.2,zorder=1,color='dimgray',alpha=0.7)
    cb = plt.colorbar(im,cax=cax,orientation='horizontal',ticklocation='bottom')
    cb.minorticks_on()
    cb.ax.tick_params(labelsize=10)
    cax.set_title(clabel+grad_unit, fontsize=10)
    # 编号标记
    ax.text(0.02,0.03, fignum, fontsize=11, c='k', weight='bold', transform=ax.transAxes, va='bottom', ha='left')
    #
    fig1.savefig('/Users/vnohhf/Documents/Python/LAMOST_dust_disk/figure/2.1.'+str(n)+'_spatial_distribution_of_gradients ('+figname+').png')
    # fig1.savefig('/Users/vnohhf/Documents/Python/LAMOST_dust_disk/figure/2.1.'+str(n)+'_spatial_distribution_of_gradients ('+figname+').pdf')


#%% MCMC方法 ============================================================================
#* 数据整理 ------------------------------------------------------------------------------
Rrange  = np.array([5,6,7,8,9,10,11,12,13,14])
matter  = joblib.load('RZbin.pkl')
matter.drop_duplicates(inplace=True, ignore_index=True) #由于算法原因，会有多个bin值完全相同的情况，需要把重复值去掉
# np.where((12<matter.r) & (matter.r<13) & (abs(matter.z)<1) & (5e-4<matter.value) & (matter.value<3e-3)) # 用于手动筛选outliers的语句
outliers = {'DIB':[],'Dust':[389,  387, 416,  361, 388, 402,  362]}  #人工剔除的outlier
filter   = {'DIB':[],'Dust':np.where(matter.err/matter.value>1)[0]}
for mn in ['Dust']:
    xdata,ydata,yerr,xdata_res,ydata_res,yerr_res = [[[] for i in range(len(Rrange)-1)] for _ in range(6)] #初始化
    titletext = []
    for i in range(len(Rrange)-1):
        if i <= 4:
            fitZlim = np.array([-1.2,1.2])
        else:
            fitZlim = Zlim
        R1, R2 = Rrange[i], Rrange[i+1]
        bin_ind0 = np.where((R1 < matter.r) & (matter.r < R2) &
                            (fitZlim[0] < matter.z) &  (matter.z < fitZlim[1]) & 
                            (matter.value > 0) & 
                            (~np.isnan(matter.value)))[0]
        bin_ind = np.setdiff1d(bin_ind0, np.union1d(filter[mn],outliers[mn]))
        xdata[i] = np.array(matter.z[bin_ind])
        ydata[i] = np.array(matter.value[bin_ind])
        yerr[i]  = np.array(matter.err[bin_ind])
        #fitZlim以外的也加入，后缀_res
        bin_ind_res = np.setdiff1d(np.where((matter.r > R1) & (matter.r < R2) & (~np.isnan(matter.value)))[0], bin_ind)
        if len(bin_ind_res) > 0:
            xdata_res[i] = matter.z[bin_ind_res]
            ydata_res[i] = matter.value[bin_ind_res]
            yerr_res[i]  = matter.err[bin_ind_res]
        titletext.append('R: '+str(R1)+'-'+str(R2)+' [kpc]')

#* 预设函数 ----------------------------------------------------------------------------
import corner
import emcee

def log_likelihood(theta, x, y, yerr): #似然函数
    global R_index
    p = theta
    #双sech^2函数:
    # f = a1/np.cosh(-abs(Z-Z0)/h1*1e3)**2 + a2/np.cosh(-abs(Z-Z0)/h2*1e3)**2
    model = p[0]/np.cosh(np.log(1+np.sqrt(2)) * abs(x*1e3-p[4])/p[1])**2 + p[2]/np.cosh(np.log(1+np.sqrt(2)) * abs(x*1e3-p[4])/p[3])**2 + Halo_EBV
    #双指数函数:
    # f = a1*exp(-abs(Z-Z0)/h1*1e3) + a2*exp(-abs(Z-Z0)/h2*1e3)
    # model = p[0]*np.exp(-abs(x-p[4])/p[1]*1e3) + p[2]*np.exp(-abs(x-p[4])/p[3]*1e3)
    sigma2 = yerr ** 2 
    return -0.5 * np.sum((y-model) ** 2 / sigma2 + np.log(sigma2))

def log_probability(theta, x, y, yerr): #Dust后验
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

def log_prior(theta): #Dust先验
    global R_index
    a1,h1,a2,h2,Z0 = theta
    if R_index == 0: # R=5-6
        if (0.2 < a1 < 1.2) & (10 < h1 < 160) & (0.003 < a2 < 0.5) & (70 < h2 < 500) & (-100 < Z0 < 100):
            return 0.0
    elif R_index in [1,2,3,4]: # R=6-10
        if (0.07  < a1 < 1.2) & (45 < h1 < 250) & (3e-3 < a2 < 0.2) & (160 < h2 < 600) & (-100 < Z0 < 100):
            return 0.0
    elif R_index in [5,6]: # R=10-12
        if (0.1  < a1 < 1.2) & (45 < h1 < 250) & (2e-4 < a2 < 0.2) & (200 < h2 < 1e3) & (-150 < Z0 < 150):
            return 0.0
    elif R_index == 7: # R=12-13
        if (0.005  < a1 < 0.2) & (30 < h1 < 500) & (2e-4 < a2 < 0.03) & (300 < h2 < 1e3) & (-200 < Z0 < 200):
            return 0.0
    elif R_index == 8: # R=13-14
        if (5e-4  < a1 < 0.1) & (200 < h1 < 700) & (1e-4 < a2 < 0.04) & (400 < h2 < 1.2e3) & (-300 < Z0 < 300):
            return 0.0
    return -np.inf

#* run MCMC ================================================================
#是否画Markov Chain
plot_Markov_Chain = 0
if plot_Markov_Chain: 
    fig2, ax2 = plt.subplots(5, len(Rrange)-1, figsize=(15,5), sharex=True)
#初始化
labels = ["$a_1$", "$h_1$", "$a_2$", "$h_2$", "$Z_0$"]
mcmc_par    = np.array([np.empty(len(labels)) for _ in range(len(Rrange)-1)])
par_disply  = np.array([np.empty(3*len(labels)) for _ in range(len(Rrange)-1)])
sigma_thin  = np.array([[0,0.] for _ in range(len(Rrange)-1)])
sigma_thick = np.array([[0,0.] for _ in range(len(Rrange)-1)])
sigma_whole = np.array([[0,0.] for _ in range(len(Rrange)-1)])
frac_disp   = np.array([[0,0,0.] for _ in range(len(Rrange)-1)])
flat_samples = [[] for _ in range(len(Rrange)-1)]
#开始mcmc
for i in range(len(Rrange)-1):
    # if i in [3,4,5,6]: continue 
    # if i != 9: continue 
    if i == 0:
        initial = np.array([0.7, 60, 0.1, 300, 0]) #初值
    elif i < 7:
        initial = np.array([0.5, 100, 2e-2, 400, 0]) #初值
    else:
        initial = np.array([0.05, 300, 1e-3, 900, 0]) #初值
    pos = initial + [2e-2, 10, 9e-4, 50, 20] * np.random.randn(32, len(labels))
    nwalkers, ndim = pos.shape
    R_index = i
    print('R:%.1f-%.1f'%tuple(Rrange[i:i+2]))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(xdata[i], ydata[i], yerr[i]))
    sampler.run_mcmc(pos, 
                     50000, #链长度
                     progress=True)
    tau = sampler.get_autocorr_time() #burn-in
    print('burn-in time: '+str(np.around(tau,decimals=0))+'\n')
    flat_samples[i] = sampler.get_chain(discard=300, thin=100, flat=True) #马尔可夫链采样
    
    #* 输出拟合参数 ---------------------------------------------------------------
    samples = sampler.get_chain()
    for j in range(ndim):
        mcmc = np.percentile(flat_samples[i][:, j], [16, 50, 84])
        mcmc_par[i,j] = mcmc[1] #结果
        par_disply[i,3*j:3*j+3] = [mcmc[1], np.diff(mcmc)[0], np.diff(mcmc)[1]]
        if plot_Markov_Chain: # 是否画马尔可夫链
            ax2[j,i].plot(samples[:, :, j], "C0", alpha=0.03)
            ax2[j,i].set_xlim(0, len(samples))
            ax2[j,0].set_ylabel(labels[j],{'size':12})
            ax2[j,0].minorticks_on()
    if plot_Markov_Chain: # 是否画马尔可夫链
        ax2[0,i].set_title(titletext[i],fontsize=18)  
        ax2[-1,i].set_xlabel("step number",fontsize=12) 
 
    #* 求a1*h1, a2*h2, a1*h1+a2*h2, 1/(a1*h1/a2/h2+1) 的1σ相对误差 -----------------
    a1h1  = np.percentile(flat_samples[i][:,0]*flat_samples[i][:,1], [16, 50, 84])
    a2h2  = np.percentile(flat_samples[i][:,2]*flat_samples[i][:,3], [16, 50, 84])
    whole = np.percentile(flat_samples[i][:,0]*flat_samples[i][:,1]+flat_samples[i][:,2]*flat_samples[i][:,3], [16, 50, 84])
    frac_thick = np.percentile(1/(flat_samples[i][:,0]*flat_samples[i][:,1]/flat_samples[i][:,2]/flat_samples[i][:,3]+1), [16, 50, 84])
    sigma_thin [i] = np.diff(a1h1)/a1h1[1]
    sigma_thick[i] = np.diff(a2h2)/a2h2[1]
    sigma_whole[i] = np.diff(whole)/whole[1] 
    frac_disp  [i] = np.append(frac_thick[1], np.diff(frac_thick))

    #* 画corner图 --------------------------------------------------------
    fig = plt.figure(figsize=(6,6), facecolor='w', dpi=200) #
    fig4 = corner.corner(flat_samples[i], bins=50, smooth=2, smooth1d=2, 
                         truths=mcmc_par[i], quantiles=[0.16, 0.5, 0.84], 
                         show_titles=True, title_fmt='.3g', title_kwargs={"fontsize": 7.5}, 
                         labels=labels, label_kwargs={"fontsize": 10}, labelpad=0.2, fig=fig)
    fig4.suptitle(titletext[i],fontsize=12)
    fig4.subplots_adjust(bottom=0.13,left=0.13,wspace=0,hspace=0)
    fig4.savefig('figure/2.5_MCMC corner('+str(i)+').pdf')  
    fig4.savefig('figure/2.5_MCMC corner('+str(i)+').png')
    plt.close()
if plot_Markov_Chain: # 是否画马尔可夫链
    fig2.suptitle('Markov Chain') 
    fig2.tight_layout()
    fig2.savefig('figure/2.4_Markov Chain.png',dpi=200)
    plt.close()
joblib.dump([mcmc_par,sigma_thin,sigma_thick,sigma_whole,frac_disp,par_disply],'disk_fit_result.pkl')


#%% 拟合图 ===========================================================================
# 读取单盘拟合参数
def SinglePlaneFunc(x,a,z0,h):
    Z = x
    return a * np.exp(-abs(Z-z0)/h) + Halo_EBV
Single_p = joblib.load('MAIN3_disk_fit_result.pkl')

mcmc_par,sigma_thin,sigma_thick,sigma_whole,frac_disp,par_disply = joblib.load('disk_fit_result.pkl')
# mcmc_par_APO = joblib.load('/Users/vnohhf/Documents/Python/DIB_dr16/MAIN5_disk_fit_result.pkl')[0]['Dust']
fig3, axes3 = plt.subplots(3,3, figsize=(8,8), sharey='all', facecolor='w', dpi=300)
ax3 = axes3.flatten()
for mn, signal_label, signal_unit in [['Dust', 'ΔE(B-V)/Δd','  [mag/kpc]']]:
    text_parsMCMC_left  = []
    text_parsMCMC_right = []
    for i in range(len(Rrange)-1):
    #* 散点 --------------------------------------------------------------------------   
        ax3[i].scatter(xdata[i], ydata[i], c=yerr[i]/ydata[i], s=4, alpha=1, vmin=0, vmax=1.2, cmap='gray', zorder=3)
        ax3[i].errorbar(xdata[i], ydata[i], yerr=yerr[i], ms=0, fmt='.k', ecolor='gray', elinewidth=0.3, alpha=0.8, zorder=2)
        # if len(outliers[i]) != 0:
        #     im2 = ax3[i].scatter(matter.z[outliers[i]],matter.value[outliers[i]], marker='x', c='k', s=13, alpha=0.8, edgecolors='none', zorder=2) #剔除点
        if len(xdata_res[i]) != 0:
            ax3[i].scatter(xdata_res[i], ydata_res[i], marker='X', s=6, alpha=0.6, c='gray', ec='none', zorder=2)
    #* 拟合线 ------------------------------------------------------------------------
        modelMCMC = lambda x,i: mcmc_par[i,0]/np.cosh(np.log(1+np.sqrt(2)) * abs(x*1e3-mcmc_par[i,4])/mcmc_par[i,1])**2 + \
                                mcmc_par[i,2]/np.cosh(np.log(1+np.sqrt(2)) * abs(x*1e3-mcmc_par[i,4])/mcmc_par[i,3])**2 + Halo_EBV
        xarray = np.arange(Zlim[0],Zlim[1]+0.01,0.01)
        ax3[i].plot(xarray, modelMCMC(xarray,i),  c='C0', alpha=0.9, lw=0.8)  
        ax3[i].axhline(Halo_EBV,ls='-.',c='C0', alpha=0.4, lw=0.8)
    #* APOGEE的拟合线 ----------------------------------------------------------------
        # modelMCMC_APO = lambda x,i: mcmc_par_APO[i,0]*np.exp(-abs(x-mcmc_par_APO[i,4])/mcmc_par_APO[i,1]*1e3) + mcmc_par_APO[i,2]*np.exp(-abs(x-mcmc_par_APO[i,4])/mcmc_par_APO[i,3]*1e3)
        # xarray = np.arange(Zlim[0],Zlim[1]+0.01,0.01)
        # ax3[i].plot(xarray, modelMCMC_APO(xarray,i),  c='C1', alpha=0.6)
    #* 单盘拟合线 --------------------------------------------------------------------
        ax3[i].plot(xarray, SinglePlaneFunc(xarray,*Single_p[i]),  c='C2', alpha=0.9, lw=0.5, ls='--')  
    #* 设置x、y轴参数 -----------------------------------------------------------------
        if i <= 4:
            fitZlim = np.array([-1.2,1.2])
            xticksLi = np.arange(-1, 1.5, 0.5)
        else:
            fitZlim = Zlim
            xticksLi = np.arange(-2, 2.5, 1)
        ax3[i].set_xlim(fitZlim)
        ax3[i].set_ylim([0.0005,  1.5]) 
        if ax3[i].is_last_row() :  ax3[i].set_xlabel('Z  [kpc]',{'size':12})
        if ax3[i].is_first_col():  ax3[i].set_ylabel(signal_label + signal_unit, {'size':12})
        ax3[i].set_xticks(xticksLi)
        # ax3[i].set_xticklabels(['%.1f'%_ for _ in np.arange(Zlim[0]+0.5,Zlim[1], 0.5)],rotation=40,fontsize=8)
        ax3[i].set_yscale('log')
        ax3[i].set_title(titletext[i], {'size':13})
    #* 拟合参数信息 --------------------------------------------------------------------
        text_parsMCMC_left .append("$a_1 = %.2f_{-%.2f}^{+%.2f}$\n$h_1 = %.f_{-%.f}^{+%.f}$\n$f_{thick} = %.2f_{-%.2f}^{+%.2f}$"%tuple(np.append(par_disply[i][:6],frac_disp[i])))
        text_parsMCMC_right.append('$a_2 = %.3f_{-%.3f}^{+%.3f}$\n$h_2 = %.f_{-%.f}^{+%.f}$\n$Z_0 = %.f_{-%.f}^{+%.f}$'%tuple(np.hstack([par_disply[i][6:12],par_disply[i][12:15]])))
        ax3[i].text(0.04, 0.95, text_parsMCMC_left[i],  fontsize=7, c='C0', transform=ax3[i].transAxes, va='top', linespacing = 1.4)
        ax3[i].text(0.96, 0.95, text_parsMCMC_right[i], fontsize=7, c='C0', transform=ax3[i].transAxes, va='top', ha='right', linespacing = 1.4)
    #* 美化 ---------------------------------------------------------------------------
        ax3[i].grid(True,ls=':',lw=0.2,zorder=1,color='dimgray',alpha=0.7)
        ax3[i].minorticks_on()
        # ax3[i].tick_params(right=True,labelright=True)
    fig3.align_labels()
    fig3.subplots_adjust(hspace=0.3, wspace=0, top=0.95, bottom=0.1, left=0.1, right=0.95)
    fig3.savefig('figure/2.6_disks MCMC Fitting.pdf')
    fig3.savefig('figure/2.6_disks MCMC Fitting.png')

#* 输出latex格式表格 ----------------------------------------------------------------------
if 0:
    print('\t'+r'Region (kpc) & $a_1$ (mag/kpc) & $h_1$ (pc) & $a_2$ (mag/kpc) & $h_2$ (pc) & $Z_0$ (pc) & $f_{thick}$ \\')
    print('\t'+'\midrule')
    for i in range(len(Rrange)-1): 
        print('\t' + 
            r'\multicolumn{1}{c|}{$%.f \le R\,\textless\,%.f$} & '%tuple(Rrange[i:i+2]) +
            '$%.3f_{-%.3f}^{+%.3f}$ & '%tuple(par_disply[i][0:3]) + 
            '$%.f_{-%.f}^{+%.f}$ & '%tuple(par_disply[i][3:6]) + 
            '$%.3f_{-%.3f}^{+%.3f}$ & '%tuple(par_disply[i][6:9]) + 
            '$%.f_{-%.f}^{+%.f}$ & '%tuple(par_disply[i][9:12]) + 
            '$%.f_{-%.f}^{+%.f}$ & '%tuple(par_disply[i][12:15]) + 
            r'$%.2f_{-%.2f}^{+%.2f}$ \\'%tuple(frac_disp[i]))


#%% 拟合薄盘、厚盘和整体标长 =============================================================
def log_likelihood2(theta, x, y, yerr): #似然函数
    model = theta[0]*np.exp(-x/theta[1])
    sigma2 = yerr ** 2 
    return -0.5 * np.sum((y-model) ** 2 / sigma2 + np.log(sigma2))

def log_probability2(theta, x, y, yerr): #Dust后验
    lp = log_prior2(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood2(theta, x, y, yerr)

def log_prior2(theta): #Dust先验
    a,l = theta
    if (0.05 < a < 0.5) & (2 < l < 12):
        return 0.0
    return -np.inf

for n, mn, signal_label in [[1,'Dust','E(B-V)  [mag]']]:
    R_array = (Rrange[1:]+Rrange[:-1])/2
    data_label = [[''] for _ in range(3)]

    p, pcov, perr = np.zeros((3,2)), [np.zeros((2,2)) for _ in range(3)], np.zeros((3,2))
    #* 对不同R区间的薄盘、厚盘、晕分别沿z积分
    step_size = 0.003
    Z_array = np.arange(Zlim[0],Zlim[1]+step_size,step_size)
    integral = pd.DataFrame(data=np.zeros((len(Rrange)-1,3)),columns=['Whole','Thin','Thick'])
    for i in range(len(integral)):
        integral.Thin [i] = sum(step_size * mcmc_par[i,0]/np.cosh(np.log(1+np.sqrt(2)) * abs(Z_array*1e3-mcmc_par[i,4])/mcmc_par[i,1])**2)
        integral.Thick[i] = sum(step_size * mcmc_par[i,2]/np.cosh(np.log(1+np.sqrt(2)) * abs(Z_array*1e3-mcmc_par[i,4])/mcmc_par[i,3])**2)
        integral.Whole[i] = integral.Thin[i] + integral.Thick[i]

    #* curve_fit拟合三种盘
    for j,comp,integral_err, in zip([0,1,2],['Whole','Thin','Thick'],[sigma_whole,sigma_thin,sigma_thick]):
        x,y,yerr = R_array, integral[comp], integral[comp] * (integral_err[:,0]+integral_err[:,1])/2
        # 参与拟合的点的序号 [全部参与]
        if j in [1,2]: 
            fit_idx = [2,3,4,5,6]
            fit_idx = np.array(integral.index)
        else:
            fit_idx = np.array(integral.index)
        exp_func = lambda x,a,b: a*np.exp(-x/b)
        p[j], pcov[j] = curve_fit(exp_func, x[fit_idx], y[fit_idx], sigma=yerr[fit_idx])
        perr[j] = np.sqrt(np.diag(pcov[j]))

#* run MCMC ================================================================
for n, mn, signal_label in [[1,'Dust','E(B-V)  [mag]']]:
    sl_par,sl_disp = dict(),dict()
    #初始化
    labels = ["$a$", "$l$"]
    sl_par    = np.array([np.empty(len(labels)) for _ in range(3)])
    sl_disp  = np.array([np.empty(3*len(labels)) for _ in range(3)])
    flat_samples = [[] for _ in range(3)]
    initial = p #初值
    #开始mcmc
    for i,comp,integral_err, in zip([0,1,2],['Whole','Thin','Thick'],[sigma_whole,sigma_thin,sigma_thick]):
        x,y,yerr = R_array, integral[comp], integral[comp] * (integral_err[:,0]+integral_err[:,1])/2
        pos = initial[i] + [0.02, 0.1] * np.random.randn(32, len(labels))
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability2, args=(x,y,yerr))
        sampler.run_mcmc(pos, 
                         6000, #链长度
                         progress=True)
        tau = sampler.get_autocorr_time() #burn-in
        print('burn-in time: '+str(np.around(tau,decimals=0))+'\n')
        flat_samples[i] = sampler.get_chain(discard=100, thin=100, flat=True) #马尔可夫链采样
        
        #* 输出拟合参数 ------------------------------------------------------
        samples = sampler.get_chain()
        for j in range(ndim):
            mcmc = np.percentile(flat_samples[i][:, j], [16, 50, 84])
            sl_par[i,j] = mcmc[1] #结果
            sl_disp[i,3*j:3*j+3] = [mcmc[1], np.diff(mcmc)[0], np.diff(mcmc)[1]]

        #* 画corner图 --------------------------------------------------------
        fig4 = corner.corner(flat_samples[i], labels=labels, truths=sl_par[i], quantiles=[0.16, 0.5, 0.84], title_fmt='.2g', label_kwargs={"fontsize": 12}, title_kwargs={"fontsize": 13}, facecolor='w')
        fig4.suptitle(titletext[i],fontsize=18)
        fig4.savefig('figure/2.7_MCMC corner('+str(i)+').png',dpi=200)
        plt.close()
    
#%% 画图
fig5, ax5 = plt.subplots(figsize=(5,4), sharex=True, facecolor='w', dpi=300)
# fig5, ax5 = plt.subplots(figsize=(5,5), sharex=True, facecolor='w', dpi=300)
for n, mn, signal_label in [[1,'Dust','E(B-V)  [mag]']]:
    for j,comp,integral_err, in zip([0,1,2],['Whole','Thin','Thick'],[sigma_whole,sigma_thin,sigma_thick]):
        x,y,yerr = R_array, integral[comp], integral[comp] * (integral_err[:,0]+integral_err[:,1])/2
        para_label = ' dust disk : $%.1f_{-%.1f}^{+%.1f}$'%(sl_par[j,1], sl_disp[j,4], sl_disp[j,5])
        ax5.errorbar(x, y, yerr=np.array(integral[comp])*integral_err.T, ms=8, alpha=0.7, elinewidth=0.7, fmt='.', mec='none', label=comp+para_label+' kpc', zorder=1)
        #标记不拟合的点为空心 [暂无]
        nofit_idx  = np.setdiff1d(np.array(integral.index), fit_idx)
        ax5.scatter(x[nofit_idx], y[nofit_idx], marker='.', facecolor='w', edgecolors='C'+str(j), lw=0.2, zorder=2) 
        xarray = np.linspace(Rrange[0],Rrange[-1],100)
        ax5.plot(xarray, exp_func(xarray,*sl_par[j]),ls='-',c='C'+str(j), alpha=0.7)
        ax5.fill_between(xarray, exp_func(xarray,*(sl_par[j]-sl_disp[j,[1,4]])), exp_func(xarray,*(sl_par[j]+sl_disp[j,[2,5]])), fc='C'+str(j), alpha=0.15) #画置信区间
        ax5.legend(fontsize=10)
        ax5.set(xlabel='R  [kpc]',ylabel=signal_label, yticks=np.arange(0,0.17,0.02))
        ax5.set_xticks(np.arange(5,15))
        ax5.grid(True,ls=':',lw=0.2,zorder=1,color='dimgray')
# 编号标记
ax5.text(0.02,0.03, '(b)', fontsize=11, c='k', weight='bold', transform=ax5.transAxes, va='bottom', ha='left')
fig5.subplots_adjust(left=0.13, right=0.95, top=0.95, bottom=0.12)
fig5.savefig('figure/2.7_scale-length fit.pdf')
fig5.savefig('figure/2.7_scale-length fit.png')
joblib.dump([integral,sl_par,sl_disp], 'MAIN2_scale-length parameters.pkl')

