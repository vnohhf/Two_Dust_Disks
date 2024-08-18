.run

restore,'halodust.sav'
;contour,(data.k > (-0.01)) < 0.01,data.r,data.z,/fill,nlevels=10
ind=where(abs(result.b) ge 20)
mwrfits,result(ind),'halodust.fits'
data=[data1,data2]
;atv,result.k(14) > (-0.1) < 0.1

xx=make_array(162*2+1,81*2+1,/float)
yy=make_array(162*2+1,81*2+1,/float)
;162,81
for i=-162,162 do begin
    for j=-81,81 do begin
       if(i le 0) then xx(i+162,j+81)=i;+162
       if(i gt 0) then xx(i+162,j+81)=i;-162
       yy(i+162,j+81)=j
    endfor
endfor

for k=43,49 do begin 
     tmp=result.k(k);reverse(result.k(k),2)
     tmpm1=result.k(k-1);reverse(result.k(k-1),2)
     tmpp1=result.k(k+1);reverse(result.k(k+1),2)
     mwrfits, tmp,'halodust.d'+strmid(string(k/100.,format='(f4.2)'),2,2)+'.fits',/create
     data_tmp=xx*0.-99
     data_tmp2=data_tmp
     weight_tmp=1./result.k_err(k)/result.k_err(k)
     weight_tmpm1=1./result.k_err(k-1)/result.k_err(k-1)
     weight_tmpp1=1./result.k_err(k+1)/result.k_err(k+1)
     for i=-162,162 do begin
       if(i mod 10 eq 0) then print,'here',i
       for j=-81,81 do begin
           wcsxy2sph,xx(i+162,j+81),yy(i+162,j+81),l,b,21
           l=l-180+360
           if(l ge 360) then l=l-360.
           if(l ge -1000) then begin 
                 ind=where( abs(result.l-l)*cos(b*!pi/180) le 3.5 and abs(result.b-b) le 5. and tmp ne -99.)
                 indm1=where( abs(result.l-l)*cos(b*!pi/180) le 3.5 and abs(result.b-b) le 5. and tmpm1 ne -99.)
                 indp1=where( abs(result.l-l)*cos(b*!pi/180) le 3.5 and abs(result.b-b) le 5. and tmpp1 ne -99.)
             val=0. & weight=0
             print,n_elements([ind,indm1,indp1])
             if(ind(0) ge 0) then val=val+ total(tmp(ind)*weight_tmp(ind)) 
             if(ind(0) ge 0) then weight=weight+ total(weight_tmp(ind)) 
             if(weight gt 0) then data_tmp2(i+162,j+81)=val/weight
             if(indm1(0) ge 0) then val=val+ total(tmpm1(indm1)*weight_tmpm1(indm1)) 
             if(indm1(0) ge 0) then weight=weight+ total(weight_tmpm1(indm1)) 
             if(indp1(0) ge 0) then val=val+ total(tmpp1(indp1)*weight_tmpp1(indp1)) 
             if(indp1(0) ge 0) then weight=weight+ total(weight_tmpp1(indp1)) 
             if(weight gt 0) then data_tmp(i+162,j+81)=val/weight
           endif
       endfor
     endfor
     data_tmp2=reverse(data_tmp2,1)
     data_tmp=reverse(data_tmp,1)
     mwrfits, data_tmp,'halodust_smooth.d'+strmid(string(k/100.,format='(f4.2)'),2,2)+'.fits',/create
     mwrfits, data_tmp2,'halodust_smooth2.d'+strmid(string(k/100.,format='(f4.2)'),2,2)+'.fits',/create

endfor
files=findfile('halodust_smooth.d*.fits')
for i=0l,n_elements(files)-1 do begin 
    xx=mrdfits(files(i),0,h)
    ind=where(xx ne -99.)
    print,files(i), mean(xx(ind)), stddev(xx(ind)), robust_sigma(xx(ind))
    xx(ind)=xx(ind)*0.02/stddev(xx(ind))
    mwrfits,xx, files(i)+'2',/create
endfor
files=findfile('halodust_smooth2.d*.fits')
for i=0l,n_elements(files)-1 do begin
    xx=mrdfits(files(i),0,h)
    ind=where(xx ne -99.)
    print,files(i), mean(xx(ind)), stddev(xx(ind)), robust_sigma(xx(ind))
    xx(ind)=xx(ind)*0.02/stddev(xx(ind))
    mwrfits,xx, files(i)+'2',/create
endfor



set_plot,'ps'
device,filename='k_r_z.ps'
device,/color
loadct,4
usersymbol,'circle',/fill,size=0.5
ind=where(data.k_err gt 0 and data.k_err le 0.002 and abs(data.z) ge 1 and data.z le 20)
plot,data(ind).r,data(ind).z,psym=3,xr=[0,30],yr=[-20,20],position=[0.1,0.1,0.8,0.95],xtitle='R (kpc)',ytitle='Z (kpc)',charsize=1.5
colors=bytscl(data(ind).k,min=-0.01,max=0.01)
for i=0l,n_elements(ind)-1 do oplot,[data(ind(i)).r,data(ind(i)).r],[data(ind(i)).z,data(ind(i)).z],psym=8,color=colors(i)
colorbar,range=[-10,10],/vertical,divisions=4,charsize=1.,position=[0.88, 0.10, 0.93, 0.90]
xyouts,0.83,0.92, '(mmag/kpc)',/normal

ind=where(data.k_err gt 0 and (data.k_err/data.k le 0.3 or data.k_err le 0.005)  and abs(data.z) le 1)
plot,data(ind).r,data(ind).z,psym=3,xr=[3,22],yr=[-1,1],xstyle=1,xtitle='R (kpc)',ytitle='Z (kpc)',charsize=1.5,position=[0.1,0.1,0.8,0.95]
colors=bytscl(alog10(data(ind).k>0.001),min=-3,max=-0.5)
for i=0l,n_elements(ind)-1 do oplot,[data(ind(i)).r,data(ind(i)).r],[data(ind(i)).z,data(ind(i)).z],psym=8,color=colors(i),symsize=1.5
colorbar,range=[-3,-0.5],format='(f4.1)',/vertical,divisions=5,charsize=1.,position=[0.88, 0.10, 0.93, 0.90]
xyouts,0.83,0.92, '(mag/kpc)',/normal
device,/close_file


ind=where(data.num ge 100)
indd=where(data.num ge 100 and data.z ge 2)
plot,data(ind).z,data(ind).k,psym=3

plot,data(ind).z,data(ind).k+0.01,psym=1,/ylog,yr=[0.001,0.1],/xlog,xr=[0.1,100]  
plot,data(ind).z,data(ind).k,psym=1,yr=[-0.02,0.02],xr=[0,40]
plot, sqrt(data(indd).z*data(indd).z+data(indd).r*data(indd).r ), data(indd).k,psym=1,yr=[-0.01,0.01],xr=[0,50]  
y_2d_locus,sqrt(data(indd).z*data(indd).z+data(indd).r*data(indd).r ), data(indd).k,x,med,sigma,min=5,max=45,bin=5
oplot,x,med,psym=-1

indd=where(data.num ge 100 and data.z ge 1)                                                                     
print,median(data(indd).k)                 
;   0.00197802
indd=where(data.num ge 100 and data.z ge 2)
print,median(data(indd).k)                 
;   0.00153700
indd=where(data.num ge 100 and data.z ge 4)
print,median(data(indd).k)                 
;   0.00115097
indd=where(data.num ge 100 and data.z ge 6)
print,median(data(indd).k)                 
; -6.87041e-06

end
