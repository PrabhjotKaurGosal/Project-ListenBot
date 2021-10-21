# This code calculates the time delay between two signals received by microphones spaced apart from each other,
# using the Generalized Cross Correlation -PHAT (GCC-PHAT) and the finds the Direction of Arrival angle (azimuth angle) for the sound.

# The caluclations for DOA in this file are the same as in AudioAnalysisClass.py... This file is to help in debugging

import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import wave
import librosa as lr
from scipy import signal
from scipy.fftpack import fft,fft2, fftshift, ifft
import math
import matplotlib.pyplot as plt

 
# Computer the time delay between two signals using the Generalized Cross Correlation method: GCC-PHAT
def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)  # Fourier Transform of sig
    REFSIG = np.fft.rfft(refsig, n=n) # Fourier Transform of refsig
    R = SIG * np.conj(REFSIG) # Cross Correlation in frequency domain (This is also called the Cross Spectral Density)
    
    # The formula for GCC-PHAT in time domain is = INVERSE_FOURIER ([(SIG)*conj(REFSIG)]/ [magnitude(SIG)*magnitude(conj(REFSIG))]) 
    # That is, GCC-PHAT in tme domain = INVERSE_FOURIER(R/magnitide(R))
    cc = np.fft.irfft(R / np.abs(R), n=(interp * n)) # GCC-PHAT in time domain

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau, cc


def CalculateTimeDifferenceOfArrival(ch1,ch2,ch3, sfreq):
    ###### Calculate time delay between channel 1 and channel 2
    y1 = ch1
    y2 = ch2

    channelNames = ["ch1", "ch2"]
    # Method: Generalized Cross Correlation -PHAT (GCC-PHAT)
    tau_gcc, cc = gcc_phat(y2,y1,sfreq)
    if tau_gcc<0:
        print("The signal ", channelNames[0], " lags behind signal ", channelNames[1])
    elif tau_gcc>0:
        print("The signal ", channelNames[0], " leads signal ",channelNames[1])
    else:
        print("The lag is Zero between the two signals")
    print("The time delay between the signals ", channelNames[0], " and ", channelNames[1], " using GCC-PHAT is: ", tau_gcc, "seconds")
    T_12_gccPhat = tau_gcc

    ###### Calcualte time delay between channel 1 and channel 3
    y1 = ch1
    y2 = ch3

    channelNames = ["ch1", "ch3"]
    # Method: Generalized Cross Correlation -PHAT (GCC-PHAT)
    tau_gcc, cc = gcc_phat(y2,y1,sfreq)
    if tau_gcc<0:
        print("The signal ", channelNames[0], " lags behind signal ", channelNames[1])
    elif tau_gcc>0:
        print("The signal ", channelNames[0], " leads signal ",channelNames[1])
    else:
        print("The lag is Zero between the two signals")
    print("The time delay between the signals ", channelNames[0], " and ", channelNames[1], " using GCC-PHAT is: ", tau_gcc, "seconds")
    T_13_gccPhat = tau_gcc

    # Calculate time delay between channel 2 and channel 3
    y1 = ch2
    y2 = ch3

    channelNames = ["ch2", "ch3"]
    # Method: Generalized Cross Correlation -PHAT (GCC-PHAT)
    tau_gcc, cc = gcc_phat(y2,y1,sfreq)
    if tau_gcc<0:
        print("The signal ", channelNames[0], " lags behind signal ", channelNames[1])
    elif tau_gcc>0:
        print("The signal ", channelNames[0], " leads signal ",channelNames[1])
    else:
        print("The lag is Zero between the two signals")
    print("The time delay between the signals ", channelNames[0], " and ", channelNames[1], " using GCC-PHAT is: ", tau_gcc, "seconds")
    T_23_gccPhat = tau_gcc

    
    return T_13_gccPhat, T_23_gccPhat, T_12_gccPhat

def CalculateDOA(ch1,ch2,ch3, sfreq):
        # Call the funtion to get the time delay
    channel1 = ch1
    channel2 = ch2
    channel3 = ch3
    samplingFreq = sfreq
    d12 = 0.094
    d13 = 0.082
    d23 = 0.046
    T13_gccPhat, T23_gcc_Phat, T12_gccPhat = CalculateTimeDifferenceOfArrival(ch1,ch2,ch3, sfreq)
        #T1 = T13 or T13_gccPhat
        #T2 = T23 or T23_gccPhat
        #T3 = T12 or T12_gccPhat

    T1 = T13_gccPhat
    T2 = T23_gcc_Phat
    T3 = T12_gccPhat 

    cos_domain_check_T1 = abs(T1) * 343
    cos_domain_check_T2 = abs(T2) * 343
    cos_domain_check_T3 = abs(T3) * 343

    angle_ch1 = math.degrees(math.acos(d13/d12))
    angle_ch2 = math.degrees(math.acos(d23/d12))
    #When sound source is in Quarter 1
    if T1 > 0 and T2 > 0:

        #Use T3
        if cos_domain_check_T3 < d12:
            angle_gccPhat_T3 = math.degrees(math.acos(abs(T3) * 343 / d12))
            if T3 < 0:
                relative_angle_gccPhat = 270 - (angle_gccPhat_T3 + angle_ch1)
            else:
                relative_angle_gccPhat = angle_gccPhat_T3 + angle_ch2
        else:
            angle_gccPhat_T2 = math.degrees(math.acos(abs(T2) * 343 / d23))
            relative_angle_gccPhat = 180 - angle_gccPhat_T2 

        if (cos_domain_check_T1<d13):
            angle_gccPhat = math.degrees(math.acos(abs(T1) * 343 / d13))
            relative_angle_gccPhat = angle_gccPhat + 90 
        else:
            #Use T2
            angle_gccPhat = math.degrees(math.acos(abs(T2) * 343 / d23))
            relative_angle_gccPhat = 180-angle_gccPhat
          
        #When sound source is in Quarter 2
    elif T1 < 0 and T2 > 0:

        #Use T2
        if cos_domain_check_T2 < d23:
            angle_gccPhat_T2 = math.degrees(math.acos(abs(T2) * 343 / d23))
            relative_angle_gccPhat = (180 - angle_gccPhat_T2) * (-1)
            
        else:
            # Use T1
            angle_gccPhat_T1 = math.degrees(math.acos(abs(T1) * 343 / d13))
            relative_angle_gccPhat = (90 + angle_gccPhat_T1) * (-1)
            
        #When sound source is in Quarter 3
    elif T1 < 0 and T2 < 0:
        # Use T3
        if cos_domain_check_T3 < d12 and T3!=0:
            angle_gccPhat = math.degrees(math.acos(abs(T3) * 343 / d12))
            if T3>0:
                relative_angle_gccPhat = (angle_gccPhat-angle_ch2)*(-1)
            elif  T3<0:
                angle_ch1_plus = 90 - (angle_gccPhat- (90-angle_ch2))
                relative_angle_gccPhat = (angle_ch1_plus)*(-1)
        else:
            # Use T2
            relative_angle_gccPhat = math.degrees(math.acos(abs(T2) * 343 / d23)) * (-1)
        #print("The estimated DOA using T3 is: ", relative_angle_gccPhat )

        #When sound source is in Quarter 4
    elif T1 > 0 and T2 <= 0:

        #Use T2
        if cos_domain_check_T2 < d23:
            angle_gccPhat_T2 = math.degrees(math.acos(abs(T2) * 343 / d23))
            relative_angle_gccPhat = angle_gccPhat_T2
        
        else:
            # Use T1
            angle_gccPhat_T1 = math.degrees(math.acos(abs(T1) * 343 / d13))
            relative_angle_gccPhat = (90 - angle_gccPhat_T1)

        #relative_angle_gccPhat = relative_angle_gccPhat_T1

        #When sound source is in bordeline of Quarter 1 and Quarter 4
    elif T2==0 and T3>0:
        relative_angle_gccPhat = 90

        #When sound source is in bordeline of Quarter 1 and Quarter 2
    elif T1==0 and T3<0:
        relative_angle_gccPhat = 180

        #When sound source is in bordeline of Quarter 2 and Quarter 3
    elif T2==0 and T3<0:
        relative_angle_gccPhat = -90    

        #When sound source is in bordeline of Quarter 3 and Quarter 4
    elif T1==0 and T3>0:
        relative_angle_gccPhat = 0

    else:
        print("There is some error in GCC-PHAT!")
            
    print("The DOA based on GCC-PHAT is: ", relative_angle_gccPhat)

    return relative_angle_gccPhat
            

if __name__ == "__main__":
        #  Read the audio files
    ch1, sfreq = lr.load("ch1.wav", sr = 44100)
    ch2, sfreq = lr.load("ch2.wav", sr = 44100)
    ch3, sfreq = lr.load("ch3.wav", sr = 44100)

    CalculateDOA(ch1,ch2,ch3, sfreq)
    print(sfreq)
