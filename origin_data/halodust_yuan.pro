.run

restore,'../egaia.gac.sav'
gac=lm
restore, '../egaia.dr5.sav'
restore,'../check_teff_extin_dependence.sav'
ind=where (abs(ga.b) ge 0. and lm.snrg ge 10. and ebv.ebprp ge -9 and (ga.ebv le 0.5  or (ga.ebv le 1. and abs(ga.b) le 10 ))) ;避免高消光区域（可以不加）
; 添加excess factor判据
ebv=ebv(ind)
ga=ga(ind)
lm=lm(ind)
;remove outliers
;ind=where ( ebv.ebprp- ga.ebv*sfit_get(lm.teff,ga.ebv, kbprp_teff_ebv)) le 0.06)
;ebv=ebv(ind)
;ga=ga(ind)
;lm=lm(ind)

;y_cross_id, gac.ra,gac.dec, lm.ra,lm.dec,min_pos,min_dist
spec_id=strmid(lm.obsdate,0,4)+strmid(lm.obsdate,5,2)+strmid(lm.obsdate,8,2)+'-'+strtrim(lm.planid,0)+'-'+strmid(string(lm.spid/100.,format='(f4.2)'),2,2)+'-'+strmid(string(lm.fiberid/1000.,format='(f5.3)'),2,3)
match, spec_id, gac.spec_id,suba,subb
ind=where(ga(suba).parallax_error/ga(suba).parallax gt 0.3  and gac(subb).dist ge 10000.) ;64
;ga(suba(ind)).parallax=1000./gac(subb(ind)).dist
;ga(suba(ind)).parallax=1./gac(subb(ind)).dist
;ga(suba(ind)).parallax_error=ga(suba(ind)).parallax*0.3 
ind=where(ga.parallax_error gt 0 and ga.parallax gt 0. and 1./ga.parallax le 100. and $ 
( ga.parallax_error/ga.parallax le 0.3 or (ga.parallax_error/ga.parallax  le 0.5 and 1./ga.parallax ge 10))) ;距离误差判据
ebv=ebv(ind)
ga=ga(ind)
lm=lm(ind)

;theta=acos(randomu(100,1000)*2-1) -!pi/2
;theta=acos(findgen(1001)/1000.*2-1) -!pi/2
; 初始化
tmp=findgen(60)*0-99. 
result=replicate({l:0.,b:0.,num:0.,nums:tmp,r:tmp,z:tmp,dists:tmp,ebprp:tmp,ebprp_err:tmp,k:tmp,k_err:tmp,ebprp_mod:tmp, k_mod:tmp},180,180)

for i=0,179 do result(i,*).l=i*2
for j=0,179 do result(*,j).b= acos( (j+0.5)/180*2-1)*180/!pi-90

for i=0,179 do begin
    for j=0, 179 do begin

        lra=result(i,j).l
        ldec=result(i,j).b
        wcs_rotate,ga.l,ga.b,phi,theta,[lra,ldec],longpole=180.,latpole=ldec,theta0=90.
        ind=where(theta ge 89.5 - abs(ldec)*3./90.)  ;radius from 0.5d to 3.5d
        print,i,j,lra,ldec
;        plot, 1./ga(ind).parallax, ebv(ind).ebprp,psym=1,xr=[0,5] 
        if(n_elements(ind) ge 2)  then begin 
          y_2d_locustmp,alog10(1./ga(ind).parallax),ebv(ind).ebprp, x,med,sigma,nums,min1=-1.,max1=2.,bin1=0.05
          result(i,j).dists=10^x
          result(i,j).ebprp=med
          result(i,j).num=n_elements(ind)
          result(i,j).nums=nums
          indtmp=where(nums ge 1)
          result(i,j).ebprp_err(indtmp)=0.02/sqrt(nums(indtmp))
          result(i,j).l=median(ga(ind).l)
          result(i,j).b=median(ga(ind).b)
        endif

       ;for k=0,48 do result(i,j).z(k)=(result(i,j).dists(k)+result(i,j).dists(k+1))/2.*sin( result(i,j).b*!pi/180)
       ;for k=0,48 do result(i,j).r(k)=sqrt(  ((result(i,j).dists(k)+result(i,j).dists(k+1))/2*cos( result(i,j).b*!pi/180))^2 + 8.*8. $
       ;               +2*8.*(result(i,j).dists(k)+result(i,j).dists(k+1))/2*sin((result(i,j).l-90)*!pi/180)*cos(result(i,j).b*!pi/180))
       ;for k=0,48 do $
       ;      if(result(i,j).ebprp(k) ne 0 and result(i,j).ebprp(k+1) ne 0) then $
       ;      result(i,j).k(k)=(result(i,j).ebprp(k+1)-result(i,j).ebprp(k))/(result(i,j).dists(k+1)-result(i,j).dists(k))

             indd=where(result(i,j).ebprp gt -9.)
             if(n_elements(indd) ge 2) then begin
               for k=0,n_elements(indd) -2 do begin
               result(i,j).k(indd(k))=(result(i,j).ebprp(indd(k+1))-result(i,j).ebprp(indd(k)))/(result(i,j).dists(indd(k+1))-result(i,j).dists(indd(k)))
               result(i,j).k_err(indd(k))=sqrt( (result(i,j).ebprp_err(indd(k+1)))^2+ (result(i,j).ebprp_err(indd(k)))^2)/(result(i,j).dists(indd(k+1))-result(i,j).dists(indd(k)))
               result(i,j).z(indd(k))=(result(i,j).dists(indd(k))+result(i,j).dists(indd(k+1)))/2.*sin( result(i,j).b*!pi/180)
               result(i,j).r(indd(k))=sqrt(  ((result(i,j).dists(indd(k))+result(i,j).dists(indd(k+1)))/2*cos( result(i,j).b*!pi/180))^2 + 8.*8. $
                      +2*8.*(result(i,j).dists(indd(k))+result(i,j).dists(indd(k+1)))/2*sin((result(i,j).l-90)*!pi/180)*cos(result(i,j).b*!pi/180))
               endfor
             endif
     endfor
     if (i mod 10 eq 0) then save,result,filename='halodust_tmp.sav'
endfor

save,result,filename='halodust.sav'

ind=where(result.k gt -90 and result.k ne 0 and result.k_err gt 0)

rr=(result.r)(ind)
zz=((result.z)(ind))
kk=(result.k)(ind)
kk_err=(result.k_err)(ind)
data=replicate({r:0.,z:0.,k:-9.,k_mean:-9.,k_err:-9.,sigmak:-0.,num:0.},100,60)
for i=0,99 do begin
    for j=0,59 do begin
        data(i,j).r=i+1
        data(i,j).z=10^( -1.+j*0.05)
        ind=where(rr ge data(i,j).r -0.5 and rr le data(i,j).r +0.5 and zz ge 10^( -1.025+j*0.05) and zz le 10^( -0.975+j*0.05) )
        if(ind(0) ge 0) then begin
           data(i,j).r=median(rr(ind))
           data(i,j).z=median(zz(ind))
           data(i,j).k=median(kk(ind))
           data(i,j).k_mean=total (kk(ind)/kk_err(ind)/kk_err(ind))/total(1./kk_err(ind)/kk_err(ind))
           data(i,j).num=n_elements(ind);median(kk(ind))
           data(i,j).sigmak=robust_sigma(kk(ind))
           data(i,j).k_err=sqrt(total(kk_err(ind)*kk_err(ind)))/n_elements(ind)
        endif
     endfor
endfor
data1=data
data=replicate({r:0.,z:0.,k:-9.,k_mean:-9.,k_err:-9.,sigmak:-0.,num:0.},100,60)
for i=0,99 do begin
    for j=0,59 do begin
        data(i,j).r=i+1
        data(i,j).z=-1*10^( -1.+j*0.05)
        ind=where(rr ge data(i,j).r -0.5 and rr le data(i,j).r +0.5 and zz ge -1*10^( -0.975+j*0.05) and zz le -1*10^( -1.025+j*0.05) )
        if(ind(0) ge 0) then begin
           data(i,j).r=median(rr(ind))
           data(i,j).z=median(zz(ind))
           data(i,j).k=median(kk(ind))
           data(i,j).k_mean=total (kk(ind)/kk_err(ind)/kk_err(ind))/total(1./kk_err(ind)/kk_err(ind))
           data(i,j).num=n_elements(ind);median(kk(ind))
           data(i,j).sigmak=robust_sigma(kk(ind))
           data(i,j).k_err=sqrt(total(kk_err(ind)*kk_err(ind)))/n_elements(ind)
        endif
     endfor
endfor
data2=data

save,result,data1,data2,filename='halodust.sav'

end
