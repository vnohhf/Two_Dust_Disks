.run

restore,'../halodust.sav'
data=[data1,data2]

set_plot,'ps'
device,/color
tvlct,255,0,0,1
tvlct,0,0,255,2
usersymbol,'circle',/fill,size=0.3

ind=where(abs(data.z) le 1.0 and data.r ne 1. and abs(data.z)  ge 0.1 and $
data.k_err gt 0. and (data.k_err/data.k le 0.2 or data.k_err le 0.005) and data.r le 13.5 and data.r ge 6.5)
x=replicate({r:0.,z:0.},n_elements(ind))
x.r=(data.r)(ind) & x.z=(data.z)(ind)
y=(data.k)(ind)-0.005*10^(-1*sqrt(x.r*x.r+x.z*x.z)/23.3) ; > 0.0001
yerr=(data.k_err)(ind)

 parinfo=replicate({value:0.D, fixed:0, limited:[0,0], $
                       limits:[0.D,0],mpside:2,tied:''}, 8)
          weights=1./yerr/yerr ;make_array(n_elements(y),value=1)
          parinfo(0).value=0.2
          parinfo(1).value=10.0
          parinfo(2).value=0.07
          parinfo(3).value=0.023
          parinfo(4).value=0.05
          parinfo(5).value=10.
          parinfo(6).value=0.5
          parinfo(4).limited=[1,1] & parinfo(4).limits=[0.01,1] ;k2  
          parinfo(2).limited=[1,1] & parinfo(2).limits=[0.05,0.1] ;scale height 
          parinfo(6).limited=[1,1] & parinfo(6).limits=[0.1,1] ;scale height
          parinfo([1,5]).limited=[1,1] & parinfo([1,5]).limits=[1,20] ;scale length 
          parinfo(7).value=-0.02

a=mpfitfun('myfunction_2disks',x,y,parinfo=parinfo,weights=weights,yfit=yfit,covar=covar,perror=perror,/nan)
;ind=where( abs(y-yfit) le 2. *robust_sigma(y-yfit))
;x=x(ind) & y =y(ind) & weights=weights(ind)
;a=mpfitfun('myfunction_2disks',x,y,parinfo=parinfo,weights=weights,yfit=yfit,covar=covar,perror=perror,/nan)
;ind=where( abs(y-yfit) le 2. *robust_sigma(y-yfit))
;x=x(ind) & y =y(ind) & weights=weights(ind)
;a=mpfitfun('myfunction_2disks',x,y,parinfo=parinfo,weights=weights,yfit=yfit,covar=covar,perror=perror,/nan)

device,filename='fit_2disk.ps',xsize=18,ysize=10
!p.multi=[0,3,2,0,0]
plot,x.z,y,psym=8,/ylog,yr=[0.001,0.3],/ystyle,xtitle='Z (kpc)',ytitle='Dust density (mag/kpc)'
oplot,x.z,yfit,psym=8,color=1,symsize=0.8
xyouts,-0.9,0.1,'k1: '+string(a(0),format='(f5.3)'),charsize=0.5
xyouts,-0.9,0.07,'L1: '+string(a(1),format='(f4.1)'),charsize=0.5
xyouts,-0.9,0.05,'H1: '+string(a(2),format='(f5.3)'),charsize=0.5
xyouts,-0.9,0.035,'Z1: '+string(a(3),format='(f6.3)'),charsize=0.5
xyouts,0.4,0.1,'k2: '+string(a(4),format='(f5.3)'),charsize=0.5
xyouts,0.4,0.07,'L2: '+string(a(5),format='(f4.1)'),charsize=0.5
xyouts,0.4,0.05,'H2: '+string(a(6),format='(f5.3)'),charsize=0.5
xyouts,0.4,0.035,'Z2: '+string(a(7),format='(f6.3)'),charsize=0.5

plot,x.z,y-yfit,xr=[-1.1,1.1],xstyle=1,psym=8,xtitle='Z (kpc)',ytitle='Residual (mag/kpc)',yr=[-0.04,0.04]
oplot,[-2,2],[0,0]
ind=where(abs(data.z) le 1. and abs(data.z)  ge 0.1 and $
data.k_err gt 0. and (data.k_err/data.k le 0.2 or data.k_err le 0.005) )
;oplot,(data.z)(ind),(data.k)(ind)-0.005*10^(-1*sqrt((data.r)(ind)*(data.r)(ind)+(data.z)(ind)*(data.z)(ind))/23.3) - $ 
;    a(0)*exp(-1*((data.r)(ind)-8.)/a(1))*exp(-1*abs((data.z)(ind)-a(3))/a(2)) ,color=1,psym=8,symsize=0.75
oplot,(data.z)(ind),(data.k)(ind)-0.005*10^(-1*sqrt((data.r)(ind)*(data.r)(ind)+(data.z)(ind)*(data.z)(ind))/23.3) - $ 
     myfunction_2disks(data(ind),a) ,color=1,psym=8,symsize=0.75
plot,x.r,y-yfit,psym=8,xr=[5,16],xtitle='R (kpc)',ytitle='Residual (mag/kpc)',yr=[-0.03,0.03]
oplot,(data.r)(ind),(data.k)(ind)-0.005*10^(-1*sqrt((data.r)(ind)*(data.r)(ind)+(data.z)(ind)*(data.z)(ind))/23.3) - $ 
     myfunction_2disks(data(ind),a) ,color=1,psym=8,symsize=0.75
oplot,[0,20],[0,0]
xyouts,4.5,0.03,'Resi.: '+string(robust_sigma(y-yfit),format='(f6.4)'),charsize=0.5
chi2=total((y-yfit)^2*weights)/(n_elements(y)-8.)
indtmp=where( abs(y-yfit) le 3* robust_sigma(y-yfit))
chi22=total( ((y-yfit)^2*weights)(indtmp))/(n_elements(indtmp)-8.)
xyouts,4.5,0.02,'Chi2: '+string(chi22,format='(f4.1)'),charsize=0.5

parinfo([3,7]).fixed=1
parinfo([3,7]).value=0.0;25
a=mpfitfun('myfunction_2disks',x,y,parinfo=parinfo,weights=weights,yfit=yfit,covar=covar,perror=perror,/nan)
;ind=where( abs(y-yfit) le 2. *robust_sigma(y-yfit))
;x=x(ind) & y =y(ind) & weights=weights(ind)
;a=mpfitfun('myfunction_2disks',x,y,parinfo=parinfo,weights=weights,yfit=yfit,covar=covar,perror=perror,/nan)
;ind=where( abs(y-yfit) le 2. *robust_sigma(y-yfit))
;x=x(ind) & y =y(ind) & weights=weights(ind)
;a=mpfitfun('myfunction_2disks',x,y,parinfo=parinfo,weights=weights,yfit=yfit,covar=covar,perror=perror,/nan)
plot,x.z,y,psym=8,/ylog,yr=[0.001,0.3],/ystyle,xtitle='Z (kpc)',ytitle='Dust density (mag/kpc)'
oplot,x.z,yfit,psym=8,color=1,symsize=0.8
xyouts,-0.9,0.1,'k1: '+string(a(0),format='(f5.3)'),charsize=0.5
xyouts,-0.9,0.07,'L1: '+string(a(1),format='(f4.1)'),charsize=0.5
xyouts,-0.9,0.05,'H1: '+string(a(2),format='(f5.3)'),charsize=0.5
xyouts,-0.9,0.035,'Z1: '+string(a(3),format='(f5.3)'),charsize=0.5
xyouts,0.4,0.1,'k2: '+string(a(4),format='(f5.3)'),charsize=0.5
xyouts,0.4,0.07,'L2: '+string(a(5),format='(f4.1)'),charsize=0.5
xyouts,0.4,0.05,'H2: '+string(a(6),format='(f5.3)'),charsize=0.5
xyouts,0.4,0.035,'Z2: '+string(a(7),format='(f5.3)'),charsize=0.5

plot,x.z,y-yfit,xr=[-1.1,1.1],xstyle=1,psym=8,xtitle='Z (kpc)',ytitle='Residual (mag/kpc)',yr=[-0.04,0.04]
oplot,[-2,2],[0,0]
ind=where(abs(data.z) le 1. and abs(data.z)  ge 0.1 and $
data.k_err gt 0. and (data.k_err/data.k le 0.2 or data.k_err le 0.005) )
oplot,(data.z)(ind),(data.k)(ind)-0.005*10^(-1*sqrt((data.r)(ind)*(data.r)(ind)+(data.z)(ind)*(data.z)(ind))/23.3) - $
     myfunction_2disks(data(ind),a) ,color=1,psym=8,symsize=0.75
plot,x.r,y-yfit,psym=8,xr=[5,16],xtitle='R (kpc)',ytitle='Residual (mag/kpc)',yr=[-0.03,0.03]
oplot,(data.r)(ind),(data.k)(ind)-0.005*10^(-1*sqrt((data.r)(ind)*(data.r)(ind)+(data.z)(ind)*(data.z)(ind))/23.3) - $
     myfunction_2disks(data(ind),a) ,color=1,psym=8,symsize=0.75
oplot,[0,20],[0,0]
xyouts,4.5,0.03,'Resi.: '+string(robust_sigma(y-yfit),format='(f6.4)'),charsize=0.5
chi2=total((y-yfit)^2*weights)/(n_elements(y)-7.)
indtmp=where( abs(y-yfit) le 3* robust_sigma(y-yfit))
chi22=total( ((y-yfit)^2*weights)(indtmp))/(n_elements(indtmp)-8.)
xyouts,4.5,0.02,'Chi2: '+string(chi22,format='(f4.1)'),charsize=0.5


device,/close_file

;exit
device,filename='fit_2disks.ps',xsize=18,ysize=15
!p.multi=[0,3,3,0,0]
ind=where(abs(data.z) le 1.0 and abs(data.z)  ge 0.1 and $
data.k_err gt 0. and data.k gt 0 and (data.k_err/data.k le 0.3 or data.k_err le 0.005) )
x=replicate({r:0.,z:0.},n_elements(ind))
x.r=(data.r)(ind) & x.z=(data.z)(ind)
y=(data.k)(ind)-0.005*10^(-1*sqrt(x.r*x.r+x.z*x.z)/23.3) ; > 0.0001
yerr=(data.k_err)(ind)
;yerr=yerr > median(yerr)
weights=1./yerr/yerr;make_array(n_elements(y),value=1)

 parinfo=replicate({value:0.D, fixed:0, limited:[0,0], $
                       limits:[0.D,0],mpside:2,tied:''}, 8)
          weights=1./yerr/yerr ;make_array(n_elements(y),value=1)
          parinfo(0).value=0.8
          parinfo([1,5]).value=100000.0
         parinfo([1,5]).fixed=1
          parinfo(2).value=0.08
          parinfo(3).value=0.023
         parinfo(0).limited=[1,1] ;0.023
         parinfo(0).limits=[0.4,1.5] ;0.023
         parinfo(4).value=0.03
         parinfo(6).value=0.3
        parinfo(4).limited=[1,1] & parinfo(4).limits=[0.01,1] ;k2  
        parinfo(2).limited=[1,1] & parinfo(2).limits=[0.04,0.1] ;scale height 
        parinfo(6).limited=[1,1] & parinfo(6).limits=[0.15,1] ;scale height 
        parinfo(3).limited=[1,1] & parinfo(3).limits=[-0.03,0.03] ;zsun
        parinfo(7).value=0.023
        parinfo(7).limited=[1,1] & parinfo(7).limits=[-0.15,0.15] ;zsun 

for i=6,14 do begin
  parinfo([3,7]).fixed=0
  ind1=where(x.r ge i-0.5 and x.r le i+0.5)
  xx=x(ind1)
  yy=y(ind1)
  weightss=weights(ind1)
  a1=mpfitfun('myfunction_2disks',xx,yy,parinfo=parinfo,weights=weightss,yfit=yfit,covar=covar,perror=perror,/nan)
  ;ind1=where( abs(yy-yfit) le 2. *robust_sigma(yy-yfit))
  ;xx=xx(ind1) & yy =yy(ind1) & weightss=weightss(ind1)
  ;a1=mpfitfun('myfunction_2disks',xx,yy,parinfo=parinfo,weights=weightss,yfit=yfit,covar=covar,perror=perror,/nan)
  plot,xx.z,yy,psym=8,/ylog,yr=[0.001,0.3],ystyle=1,xtitle='Z (kpc)',ytitle='Dust density (mag/kpc)', $ 
       title='R='+string(i,format='(I2)')+'kpc',xr=[-1,1] 
  ztmp=findgen(201)/100.-1
  oplot,ztmp, a1(0)*exp(-1*(i-8.)/a1(1))*exp(-1*abs(ztmp-a1(3))/a1(2))+a1(4)*exp(-1*(i-8.)/a1(5))*exp(-1*abs(ztmp-a1(3))/a1(6)),color=1
  xyouts,-0.95,0.2,'k1: '+string(a1(0),format='(f5.3)'),charsize=0.5,color=1
  xyouts,-0.95,0.12,'H1:  '+string(a1(2),format='(f5.3)'),charsize=0.5,color=1
  xyouts,-0.95,0.08,'Z1: '+string(a1(3),format='(f6.3)'),charsize=0.5,color=1
  xyouts,-0.95,0.05,'k2: '+string(a1(4),format='(f5.3)'),charsize=0.5,color=1
  xyouts,-0.95,0.03,'H2:  '+string(a1(6),format='(f5.3)'),charsize=0.5,color=1
  xyouts,-0.95,0.02,'Z2: '+string(a1(7),format='(f6.3)'),charsize=0.5,color=1

  parinfo([3,7]).fixed=1
  parinfo([3,7]).value=0.0;25
;  if(i eq 8) then parinfo(0).fixed=1 ;  weights( where(yy ge 0.05))=weights( where(yy ge 0.05))*100.
;  if(i eq 8) then parinfo(0).value=0.6 ;  weights( where(yy ge 0.05))=weights( where(yy ge 0.05))*100.
  ;ind1=where(x.r eq i)
  ;xx=x(ind1)
  ;yy=y(ind1)
  ;weightss=weights(ind1)
  a1=mpfitfun('myfunction_2disks',xx,yy,parinfo=parinfo,weights=weightss,yfit=yfit,covar=covar,perror=perror,/nan)
  ;ind1=where( abs(yy-yfit) le 2. *robust_sigma(yy-yfit))
  ;xx=xx(ind1) & yy =yy(ind1) & weightss=weightss(ind1)
  ;a1=mpfitfun('myfunction_2disks',xx,yy,parinfo=parinfo,weights=weightss,yfit=yfit,covar=covar,perror=perror,/nan)
  oplot,ztmp, a1(0)*exp(-1*(i-8.)/a1(1))*exp(-1*abs(ztmp-a1(3))/a1(2))+a1(4)*exp(-1*(i-8.)/a1(5))*exp(-1*abs(ztmp-a1(3))/a1(6)),color=2
  xyouts,0.5,0.2,'k1: '+string(a1(0),format='(f5.3)'),charsize=0.5,color=2
  xyouts,0.5,0.12,'H:  '+string(a1(2),format='(f5.3)'),charsize=0.5,color=2
  xyouts,0.5,0.08,'Z1: '+string(a1(3),format='(f6.3)'),charsize=0.5,color=2
  xyouts,0.5,0.05,'k2: '+string(a1(4),format='(f5.3)'),charsize=0.5,color=2
  xyouts,0.5,0.03,'H2:  '+string(a1(6),format='(f5.3)'),charsize=0.5,color=2
  xyouts,0.5,0.02,'Z2: '+string(a1(7),format='(f6.3)'),charsize=0.5,color=2

endfor


device,/close_file
end

