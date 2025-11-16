#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Based on a Matlab code from 2022, based on an earlier DataGraph file

Created on Sun Dec 29 09:06:37 2024

@author: Roland Kádár
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy import ndimage
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.special import legendre
from scipy import integrate
from scipy.signal import savgol_filter
#matplotlib.rcParams['figure.figsize'] = (5, 10)
from itertools import count
import random
from moepy import lowess, eda
#import _Main_custom_LAOS_PLI_SAXS_v1 as main

def P2468OAS_v3_3(
    Angles,
    File_part_,
    lims,
    smoo_,
    sigma_zero,
    plot=True,
    preview_limit=1000,
    save_prefix=None,
    window_title=None,
):
    raw_input = np.array(File_part_, copy=True)
    File = raw_input.copy()
    lo_li = int(lims[0])
    up_li = int(lims[1])
    #preview_limit=1000
    
    #Initializations and corrections
    #Baseline = 0
    #File = File-Baseline
    Threshold = 0.0001
    [DataM,DataN] =  np.shape(File)
    #XRaww = File[:,0]*2*np.pi/DataM
    #File=File[:,1:DataN]
    #File=File[:,0:DataN]
    #DataN=DataN-1
    #print(DataM)
    Tot_ang=DataM
    UnitAngleRad = 2*np.pi/DataM
    XRaww=Angles
    
    
    Half = int(DataM/2)
    Quarter = int(Half/2)
    #print(Quarter)
            
    DataShift=np.zeros((DataM,DataN))
    DataB=np.zeros((DataM,DataN))
    DataShift=File.copy()
    #DataShift[0:Quarter,:]=File[Quarter:Half,:]
    #DataShift[Half:Tot_ang,:]=DataShift[0:Half,:]
    #DataShift[Half:Tot_ang, :] = DataShift[Half-1::-1, :] #special case. for Nguyen's spikes
    DataShift[0:Half,:]=File[Half:Tot_ang,:]               #FLIPS the side it is analyzing        
    DataIn_=np.zeros((DataM,DataN))
    DataIn=DataShift
    smooth=np.zeros((DataM,DataN))
    MAXX=np.zeros((DataN))
    Index_MAXX=np.zeros((DataN))
    smooth_bak=np.zeros((DataM,DataN))
    Fit_Lor=np.zeros((int(Quarter),DataN))
    Fit_L0146=np.zeros((int(Quarter),DataN))
    Fit_L0123456=np.zeros((int(Quarter),DataN))
    FIT=np.zeros((int(Quarter),DataN))
    Fit_G = np.zeros((int(Quarter),DataN))
    Integration_denominator=np.zeros(DataN)
    Integration_numerator_P2=np.zeros(DataN)
    Integration_numerator_P4=np.zeros(DataN)
    Integration_numerator_P6=np.zeros(DataN)
    AvgP2=np.zeros(DataN)
    AvgP4=np.zeros(DataN)
    AvgP6=np.zeros(DataN)
    HoP=np.zeros(DataN)
    P2=np.zeros(DataN)
    P4=np.zeros(DataN)
    P6=np.zeros(DataN)
    Index_MAXX_store=np.zeros(DataN)
    cmap=plt.get_cmap('turbo')
    #Y=[np.zeros((int(Half/2),DataN));]
    
    #Fit=np.zeros((6,DataN))
    
    #Smoothing
    DataB=DataShift
    #DataShift = File

    if smoo_>0:
        for i in range(0,DataN):
            #DataB[:,i] = pd.DataFrame(File[:,i])
            #DataB[:,i].replace(0, np.nan, inplace=True)
            #if i<35: 
            #    Frac=0.01
            #else: Frac=0.01
            lowess_model = lowess.Lowess()
            #smooth = sm.nonparametric.lowess(exog=XRaww, endog=DataB[:,i], frac=0.04, it=1000)
            lowess_model.fit(XRaww, DataB[:,i], frac=smoo_, robust_iters=0)
            smooth = lowess_model.predict(XRaww)
            #smooth = savgol_filter(DataB[:,i], window_length=10, polyorder=2, deriv=1)
            #smooth[:,1] = DataB[:,i]
            smooth_bak[:,i]=smooth#[:,1]
            MAXX[i] = smooth[lo_li:up_li].max(axis=0)
            Index_MAXX[i] = int(smooth[lo_li:up_li].argmax(axis=0))+lo_li
            Index_MAXX_store[i]=((Index_MAXX[i])*360/DataM)-90
            Index_MAXX_store[i] = ((Index_MAXX_store[i] + 90) % 180) - 90
            #    Index_MAXX_store[i]=Index_MAXX_store[i]+360
            #print(np.shape(Index_MAXX))
            DataNorm=smooth_bak

     
         
    #smooth_bak=DataB
    
    #DataNorm=DataB #meaning, we determine the max on smootheed data but we process raw data for Legendre fitting
    #DataNorm=DataShift
    
    # === Combined overview figure (optional) ===
    fig = None
    ax_preview = None
    ax_pvals = None
    ax_hop = None
    preview_x_min = np.inf
    preview_x_max = -np.inf
    preview_y_min = np.inf
    preview_y_max = -np.inf
    if plot:
        fig = plt.figure(figsize=(13.0, 8.5))
        if window_title and hasattr(fig.canvas.manager, "set_window_title"):
            fig.canvas.manager.set_window_title(window_title)
        gs = fig.add_gridspec(
            4,
            3,
            height_ratios=[1.0, 0.7, 1.0, 1.0],
            width_ratios=[1.0, 1.0, 1.0],
            wspace=0.35,
            hspace=0.55,
        )

        shared_cmap = plt.get_cmap("turbo")
        color_cycle = shared_cmap(np.linspace(0, 1, max(10, DataN)))

        ax_heat = fig.add_subplot(gs[0, 0])
        heat_data = np.flip(raw_input, axis=0)
        im = ax_heat.imshow(heat_data, aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        ax_heat.set_title("Raw Input Data")
        ax_heat.set_xlabel("Frame")
        ax_heat.set_ylabel("Angle idx")

        ax_raw = fig.add_subplot(gs[0, 1])
        ax_raw.set_prop_cycle(color=color_cycle)
        ax_raw.plot(raw_input, linewidth=0.4)
        ax_raw.set_title("Raw traces")
        ax_raw.set_xlabel("Frame")

        ax_shift = fig.add_subplot(gs[0, 2])
        ax_shift.set_prop_cycle(color=color_cycle)
        ax_shift.plot(DataB, linewidth=0.4)
        ax_shift.set_title("Data after shifting")
        ax_shift.set_xlabel("Frame")

        ax_smooth = fig.add_subplot(gs[1, :])
        ax_smooth.set_prop_cycle(color=color_cycle)
        ax_smooth.plot(smooth_bak, linewidth=0.4, zorder=1)
        ax_smooth.scatter(
            Index_MAXX,
            MAXX,
            s=10,
            marker="o",
            zorder=2,
            facecolor="none",
            edgecolor="k",
            linewidth=0.4,
        )
        ax_smooth.set_title("Smooth data with maxima")
        ax_smooth.set_xlabel("Angle idx")

        ax_pvals = fig.add_subplot(gs[2:, 0])
        ax_pvals.set_ylabel("P2, P4, P6")
        ax_pvals.set_xlabel("Step index")
        ax_pvals.set_box_aspect(1)

        ax_preview = fig.add_subplot(gs[2:, 1])
        ax_preview.set_title("P02468 preview (data vs fit)")
        ax_preview.set_xlabel("Angle [deg]")
        ax_preview.set_ylabel("Norm. intensity")
        ax_preview.set_box_aspect(1)

        ax_hop = fig.add_subplot(gs[2:, 2])
        ax_hop.set_title("HoP vs step")
        ax_hop.set_xlabel("Step index")
        ax_hop.set_ylabel("HoP")
    
    def Legendre_series(x,a0,a2,a4,a6,a8):
        L0=legendre(0)
        P0=L0(np.cos(x))
        L2=legendre(2)
        P2=L2(np.cos(x))
        L4=legendre(4)
        P4=L4(np.cos(x))
        L6=legendre(6)
        P6=L6(np.cos(x))
        L8=legendre(8)
        P8=L6(np.cos(x))
        Legendre_series=a0*P0+a2*P2+a4*P4+a6*P6+a8*P8
        return Legendre_series
    
    def Legendre_series_full(x,a0,a1,a2,a3):
        L0=legendre(0)
        P0=L0(np.cos(x))
        L1=legendre(1)
        P1=L1(np.cos(x))
        L2=legendre(2)
        P2=L2(np.cos(x))
        L3=legendre(3)
        P3=L3(np.cos(x))
        L4=legendre(4)
        P4=L4(np.cos(x))
        L5=legendre(5)
        P5=L5(np.cos(x))
        L6=legendre(6)
        P6=L6(np.cos(x))
        #Legendre_series_full=a0*P0+a1*P1+a2*P2+a3*P3+a4*P4+a5*P5+a6*P6
        Legendre_series_full=a0*P0+a1*P1+a2*P2+a3*P3
        return Legendre_series_full
    
    def Lorentzian_Gaussian(x,a1,Ic2,xc2,omega12_2,a2,Ic1,xc1,omega12_1,):
        LG=a1*(Ic2*(1+(np.sqrt(2)-1)*((x-xc2)/omega12_2)**2)**(-2))+a2*(Ic1*np.e**(-np.log(2)*((x-xc1)/omega12_1)**2))
        return LG
    
    def Gaussian(x,Ic1,xc1,omega12_1,):
        G=Ic1*np.e**(-np.log(2)*((x-xc1)/omega12_1)**2)
        return G
    
    color_map = shared_cmap if plot else plt.get_cmap('turbo')
    colors = color_map(np.linspace(0, 1, DataN))

    for j in range(0,DataN):    
          #print(j)
          #if Index_MAXX[j] - Quarter <0:
          #    Y=DataNorm[int(Index_MAXX[j]+Half-Quarter):int(Index_MAXX[j]+Half+Quarter),j]/DataNorm[int(Index_MAXX[j]+Half-Quarter):int(Index_MAXX[j]+Half+Quarter),j].max(axis=0)
          #else:
          #    Y=DataNorm[int(Index_MAXX[j]-Quarter):int(Index_MAXX[j]+Quarter),j]/DataNorm[int(Index_MAXX[j]-Quarter):int(Index_MAXX[j]+Quarter),j].max(axis=0)
 
          if Index_MAXX[j] < Quarter:
                start_idx = int(Index_MAXX[j])
                stop_idx = int(Index_MAXX[j]+Quarter)
                seg = DataNorm[start_idx:stop_idx, j]
                Y = seg / seg.max(axis=0)
          else:   
                start_idx = int(Index_MAXX[j]-Quarter)
                stop_idx = int(Index_MAXX[j])
                seg = DataNorm[start_idx:stop_idx, j]
                Y = seg / seg.max(axis=0)
                Y = np.flip(Y)
          XRaw1 = np.arange(0,np.size(Y),1)     
          X=(XRaw1)*np.pi/Half   
          print(np.size(X))
          print(np.size(Y))
          
         
          #print(Legendre_series(X,coeff[0],coeff[1],coeff[2],coeff[3]))
          
          sigma = np.ones(len(X))
          #sigma[0] = 0.05
          sigma[0] = sigma_zero
          #sigma[1] = 0.05
          try:
              parametersL, _ = curve_fit(Lorentzian_Gaussian,X,Y, sigma=sigma)
              #print(Y)
              coeffL = parametersL
              Fit_Lor[:,j]=Lorentzian_Gaussian(X, coeffL[0], coeffL[1], coeffL[2], coeffL[3], coeffL[4], coeffL[5], coeffL[6], coeffL[7])
              FIT[:,j] = Fit_Lor[:,j]
          except RuntimeError:
              print("Lorentzian-Gaussian fit failed; switching to Legendre series")
              #popt, pcov = curve_fit(model_func, x, y, p0=(0.1 ,1e-3, 0.1), sigma=sigma)
              parameters, _ = curve_fit(Legendre_series,X,Y, sigma=sigma)
              coeff = parameters
              Fit_L0146[:,j]=Legendre_series(X,coeff[0],coeff[1],coeff[2],coeff[3],coeff[4])
              FIT[:,j] = Fit_L0146[:,j]
          #print(coeffL)
          
          #parametersG, _ = curve_fit(Gaussian,X,Y, sigma=sigma)
          #coeffG = parametersG
          #Fit_G[:,j]=Legendre_series(X,coeff[0],coeff[1],coeff[2]) 
          
          #FIT[:,j]=Fit_G;
          
          FIT[:,j]/FIT[:,j].max(axis=0)
          deg=6
          parameters_full=np.polynomial.legendre.legfit(X,Y,deg)
          coeff_full=parameters_full
          #print(coeff_full)
          Fit_L0123456[:,j]=Legendre_series_full(X,coeff_full[0],coeff_full[1],coeff_full[2],coeff_full[3])
          Int_L012=np.polynomial.legendre.legint(Y,2)
          #print(Int_L012)
          #legvander3d - FOOD FOR THOUGHT
          #for i, ax in enumerate(axs.flatten()):
          #    ax.hist(data[i])
          #    ax.set_title(f'Dataset {i+1}')
         
          
          #If the fitting moves the peak... EXTREME CASES
          #MAXX[j] = Fit_L0123456[lo_li:up_li,1].max(axis=0)
          #Index_MAXX[j] = int(Fit_L0123456[lo_li:up_li,1].argmax(axis=0))+lo_li
          #Index_MAXX_store[j]=Index_MAXX[j]
          
          #if Index_MAXX[j] < Quarter+1:
          #      Y2=Fit_L0146[int(Index_MAXX[j]):int(Index_MAXX[j]+Quarter),j]/Fit_L0146[int(Index_MAXX[j]):int(Index_MAXX[j]+Quarter),j].max(axis=0)
                #Y=np.flip(Y)
          #else:   
          #      Y2=Fit_L0146[int(Index_MAXX[j]-Quarter):int(Index_MAXX[j]),j]/Fit_L0146[int(Index_MAXX[j]-Quarter):int(Index_MAXX[j]),j].max(axis=0)
          #      Y2=np.flip(Y)  
          #print(np.size(Y))
          #XRaw2 = np.arange(0,np.size(Y2),1)     
          #X=XRaw2*np.pi/Half
          
          
          if plot and ax_preview is not None:
              print(f"Plotting preview for column {j}, len(X)={len(X)}, len(FIT[:,j])={len(FIT[:,j])}")
              color = colors[j]
              x_preview = np.degrees(X)
              preview_x_min = min(preview_x_min, np.min(x_preview))
              preview_x_max = max(preview_x_max, np.max(x_preview))
              preview_y_min = min(preview_y_min, np.min(Y))
              preview_y_max = max(preview_y_max, np.max(Y))
              ax_preview.scatter(
                  x_preview,
                  Y,
                  s=22,
                  linewidth=0.7,
                  facecolors="none",
                  edgecolors=color,
              )
              ax_preview.plot(
                  x_preview,
                  FIT[:, j],
                  color="black",
                  linewidth=1.0,
                  alpha=0.9,
              )
    
       
          Integration_denominator[j]=integrate.trapezoid(FIT[:,j]*np.sin(X),X);
          
          Integration_numerator_P2[j]=integrate.trapezoid(FIT[:,j]*np.cos(X)**2*np.sin(X),X)
          AvgP2[j]=Integration_numerator_P2[j]/Integration_denominator[j]
    
          Integration_numerator_P4[j]=integrate.trapezoid(FIT[:,j]*np.cos(X)**4*np.sin(X),X)
          AvgP4[j]=Integration_numerator_P4[j]/Integration_denominator[j];
    
          Integration_numerator_P6[j]=integrate.trapezoid(FIT[:,j]*np.cos(X)**6*np.sin(X),X)
          AvgP6[j]=Integration_numerator_P6[j]/Integration_denominator[j]
          
          HoP[j]= (3*AvgP2[j]-1)/2; 
          P2[j] = HoP[j]
          P4[j] = (35*AvgP4[j]-30*AvgP2[j]+3)/8
          P6[j] = (231*AvgP6[j]-315*AvgP4[j]+ 105*AvgP2[j]-5)/16;
          #P8(j) = 
    
          if HoP[j]<Threshold:
              HoP[j]=0
              P4[j]=0
              P6[j]=0
              #P8[j]=
          #print(HoP[j])
    #fig, ax = plt.subplots(1,1)
    #ax.set(xlim=(0,20), ylim=(0, 5))
    #line, = ax.plot([], [], 'r-', lw=3)

    #ani = animate(fig, X, Y animate, frames=19, interval=200, repeat=False)
    
    All_in_1 = np.zeros((DataN,4))
    
    All_in_1[:,0]=HoP
    All_in_1[:,1]=P4
    All_in_1[:,2]=P6
    All_in_1[:,3]=Index_MAXX_store

    # === Finalize plotting ===
    if plot and fig is not None:
        x_time = np.arange(DataN)
        if ax_preview is not None and np.isfinite(preview_y_min):
            span_y = max(1e-6, preview_y_max - preview_y_min)
            pad_y = span_y * 0.15
            ax_preview.set_ylim(preview_y_min - pad_y, preview_y_max + pad_y)
            if np.isfinite(preview_x_min):
                span_x = max(1e-3, preview_x_max - preview_x_min)
                pad_x = span_x * 0.1
                ax_preview.set_xlim(preview_x_min - pad_x, preview_x_max + pad_x)
        if ax_pvals is not None:
            # Keep P2 colored consistently with raw/preview traces for cross-checking
            ax_pvals.scatter(
                x_time,
                P2,
                s=12,
                c=colors[: len(x_time)],
                label="P2",
            )
            ax_pvals.scatter(x_time, P4, s=12, label="P4")
            ax_pvals.scatter(x_time, P6, s=12, label="P6")
            ax_pvals.legend(loc="upper right")
        if ax_hop is not None:
            ax_hop.scatter(x_time, HoP, s=10)
        fig.tight_layout()
        if save_prefix is not None:
            try:
                fig.savefig(f"{save_prefix}_P2468_overview.png", dpi=150)
            except Exception as e:
                print(f"Warning: could not save P2468 overview figure: {e}")

        # --- Force preview display ---
        try:
            plt.ion()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.show(block=True)
            print("Interactive preview displayed.")
        except Exception as e:
            print(f"Warning: could not display interactive preview: {e}")

    return All_in_1
