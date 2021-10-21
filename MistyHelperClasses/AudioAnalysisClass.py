# A class that extracts audio channels from a raw audio input, calculates time delay between three microphones, 
# and then calculates the DOA for the sound source 

import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import wave
import librosa as lr
from scipy import signal
import math
    

class AudioAnalysisClass(object):
    def __init__(self, audioFilePath_raw, Dist_12, Dist_13, Dist_23):
        self.audioFilePath = audioFilePath_raw
        self.Dist_12 = Dist_12
        self.Dist_13 = Dist_13
        self.Dist_23 = Dist_23
        #self.ch1 = ch1
        #self.ch2 = ch2
        #self.ch3 = ch3
        #self.sfreq = sfreq

    ####  A function that takes a raw audio as input and extracts all audio channels present in the raw audio
    #     Take Wave_read object as an input and save one of its channels into a separate .wav file.
    def save_wav_channel(self, fn, wav, channel):
        # Read data
        nch   = wav.getnchannels()
        print("The number of channels present in the audio is: ", nch)
        depth = wav.getsampwidth()
        wav.setpos(0)
        sdata = wav.readframes(wav.getnframes())

        # Extract channel data (24-bit data not supported)
        typ = { 1: np.uint8, 2: np.uint16, 4: np.uint32 }.get(depth)
        if not typ:
            raise ValueError("sample width {} not supported".format(depth))
        if channel >= nch:
            raise ValueError("cannot extract channel {} out of {}".format(channel+1, nch))
        print ("Extracting channel {} out of {} channels, {}-bit depth".format(channel+1, nch, depth*8))
        data = np.fromstring(sdata, dtype=typ)
        ch_data = data[channel::nch]

        outwav = wave.open(fn, 'w')
        outwav.setparams(wav.getparams())
        outwav.setnchannels(1)
        outwav.writeframes(ch_data.tostring())
        outwav.close()
    
    # Compute the time delay between two signals using the Generalized Cross Correlation method: GCC-PHAT
    def gcc_phat(self, sig, refsig, fs=1, max_tau=None, interp=16):
    
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

    #### A function that takes all input channels and finds the time delay/Time Difference of Arrival 
    #### between the micophones
    def CalculateTimeDifferenceOfArrival(self,ch1,ch2,ch3, sfreq):
         ###### Calculate time delay between channel 1 and channel 2
        y1 = ch1
        y2 = ch2
        sfreq = sfreq

        # Method 1: Simple Cross Correlation
        correlation = signal.correlate(y2, y1, mode="full")
        lags = signal.correlation_lags(y1.size, y2.size, mode="full")
        lag = lags[np.argmax(correlation)]

        channelNames = ["ch1", "ch2"]
        if lag<0:
            print("The signal ", channelNames[0], " lags behind signal ", channelNames[1])
        elif lag>0:
            print("The signal ", channelNames[0], " leads signal ",channelNames[1])
        else:
            print("The lag is Zero between the two signals")
        print("The time delay between the signals ", channelNames[0], " and ", channelNames[1], " using simple CC is: ", lag/sfreq, "seconds")
        T_12_cc = lag/sfreq

        # Method 2: Generalized Cross Correlation -PHAT (GCC-PHAT)
        tau, cc = self.gcc_phat(y2,y1,sfreq)
        if tau<0:
            print("The signal ", channelNames[0], " lags behind signal ", channelNames[1])
        elif tau>0:
            print("The signal ", channelNames[0], " leads signal ",channelNames[1])
        else:
            print("The lag is Zero between the two signals")
        print("The time delay between the signals ", channelNames[0], " and ", channelNames[1], " using GCC-PHAT is: ", tau, "seconds")
        T_12_gccPhat = tau

        ###### Calcualte time delay between channel 1 and channel 3
        y1 = ch1
        y2 = ch3

        # Method 1: Simple Cross Correlation
        correlation = signal.correlate(y2, y1, mode="full")
        lags = signal.correlation_lags(y1.size, y2.size, mode="full")
        lag = lags[np.argmax(correlation)]

        channelNames = ["ch1", "ch3"]
        if lag<0:
            print("The signal ", channelNames[0], " lags behind signal ", channelNames[1])
        elif lag>0:
            print("The signal ", channelNames[0], " leads signal ",channelNames[1])
        else:
            print("The lag is Zero between the two signals")
        print("The time delay between the signals ", channelNames[0], " and ", channelNames[1], " is: ", lag/sfreq, "seconds")
        T_13_cc = lag/sfreq

        # Method 2: Generalized Cross Correlation -PHAT (GCC-PHAT)
        tau, cc = self.gcc_phat(y2,y1,sfreq)
        if tau<0:
            print("The signal ", channelNames[0], " lags behind signal ", channelNames[1])
        elif tau>0:
            print("The signal ", channelNames[0], " leads signal ",channelNames[1])
        else:
            print("The lag is Zero between the two signals")
        print("The time delay between the signals ", channelNames[0], " and ", channelNames[1], " using GCC-PHAT is: ", tau, "seconds")
        T_13_gccPhat = tau

        # Calculate time delay between channel 2 and channel 3
        y1 = ch2
        y2 = ch3

        # Method 1: Simple Cross Correlation
        correlation = signal.correlate(y2, y1, mode="full")
        lags = signal.correlation_lags(y1.size, y2.size, mode="full")
        lag = lags[np.argmax(correlation)]

        channelNames = ["ch2", "ch3"]
        if lag<0:
            print("The signal ", channelNames[0], " lags behind signal ", channelNames[1])
        elif lag>0:
            print("The signal ", channelNames[0], " leads signal ",channelNames[1])
        else:
            print("The lag is Zero between the two signals") 
        print("The time delay between the signals ", channelNames[0], " and ", channelNames[1], " is: ", lag/sfreq, "seconds")
        T_23_cc = lag/sfreq

        # Method 2: Generalized Cross Correlation -PHAT (GCC-PHAT)
        tau, cc = self.gcc_phat(y2,y1,sfreq)
        if tau<0:
            print("The signal ", channelNames[0], " lags behind signal ", channelNames[1])
        elif tau>0:
            print("The signal ", channelNames[0], " leads signal ",channelNames[1])
        else:
            print("The lag is Zero between the two signals")
        print("The time delay between the signals ", channelNames[0], " and ", channelNames[1], " using GCC-PHAT is: ", tau, "seconds")
        T_23_gccPhat = tau
    
        return T_13_cc,T_23_cc, T_12_cc,T_13_gccPhat, T_23_gccPhat, T_12_gccPhat

    #### A function that calculates the direction of sound (azimuth angle) using the GCC-PHAT based time delay
    def CalculateDOA(self, ch1,ch2,ch3, sfreq):
        channel1 = ch1
        channel2 = ch2
        channel3 = ch3
        samplingFreq = sfreq
        # Microphone geometry
        d12 = 0.094
        d13 = 0.082
        d23 = 0.046
        # Call the funtion to get the time delay
        T13_cc, T23_cc, T12_cc, T13_gccPhat, T23_gcc_Phat, T12_gccPhat = self.CalculateTimeDifferenceOfArrival(channel1,channel2,channel3, samplingFreq)

        #### Time delays using GCC-PHAT 
        T1 = T13_gccPhat
        T2 = T23_gcc_Phat
        T3 = T12_gccPhat 

        cos_domain_check_T1 = abs(T1) * 343
        cos_domain_check_T2 = abs(T2) * 343
        cos_domain_check_T3 = abs(T3) * 343

        angle_ch1 = math.degrees(math.acos(d13/d12))
        angle_ch2 = math.degrees(math.acos(d23/d12))

        ############  When sound source is in Quarter 1   ############
        if T1 > 0 and T2 > 0:       
            if cos_domain_check_T3 < d12 and T3!=0:
                #Use T3
                angle_gccPhat_T3 = math.degrees(math.acos(abs(T3) * 343 / d12))
                if T3 < 0:
                    relative_angle_gccPhat = 270 - (angle_gccPhat_T3 + angle_ch1)
                else:
                    relative_angle_gccPhat = angle_gccPhat_T3 + angle_ch2
            elif cos_domain_check_T2 < d23:
                #Use T2
                angle_gccPhat_T2 = math.degrees(math.acos(abs(T2) * 343 / d23))
                relative_angle_gccPhat = 180 - angle_gccPhat_T2 
            else:
                relative_angle_gccPhat = 0

        ############  When sound source is in Quarter 2   ############
        elif T1 < 0 and T2 > 0:
            if cos_domain_check_T2 < d23:
                #Use T2
                angle_gccPhat_T2 = math.degrees(math.acos(abs(T2) * 343 / d23))
                relative_angle_gccPhat = (180 - angle_gccPhat_T2) * (-1)
            
            elif cos_domain_check_T1 < d13:
                # Use T1
                angle_gccPhat_T1 = math.degrees(math.acos(abs(T1) * 343 / d13))
                relative_angle_gccPhat = (90 + angle_gccPhat_T1) * (-1)
            else:
                relative_angle_gccPhat  = 0

        ############  When sound source is in Quarter 3   ############
        elif T1 < 0 and T2 < 0:
            if cos_domain_check_T3 < d12 and T3!=0:
                # Use T3
                angle_gccPhat = math.degrees(math.acos(abs(T3) * 343 / d12))
                if T3>0:
                    relative_angle_gccPhat = (angle_gccPhat-angle_ch2)*(-1)
                elif  T3<0:
                    angle_ch1_plus = 90 - (angle_gccPhat- (90-angle_ch2))
                    relative_angle_gccPhat = (angle_ch1_plus)*(-1)
            elif(cos_domain_check_T2 < d23):
                # Use T2
                relative_angle_gccPhat = math.degrees(math.acos(abs(T2) * 343 / d23)) * (-1)
            else:
                relative_angle_gccPhat = 0

        ############  When sound source is in Quarter 4   ############
        elif T1 > 0 and T2 <= 0:
            if cos_domain_check_T2 < d23:
                 #Use T2
                angle_gccPhat_T2 = math.degrees(math.acos(abs(T2) * 343 / d23))
                relative_angle_gccPhat = angle_gccPhat_T2
            elif cos_domain_check_T1 < d13:
                # Use T1
                angle_gccPhat_T1 = math.degrees(math.acos(abs(T1) * 343 / d13))
                relative_angle_gccPhat = (90 - angle_gccPhat_T1)
            else:
                relative_angle_gccPhat = 0

        ############   When sound source is in bordeline of Quarter 1 and Quarter 4   ########
        elif T2==0 and T3>0:
            relative_angle_gccPhat = 90

        ###########    When sound source is in bordeline of Quarter 1 and Quarter 2   ########
        elif T1==0 and T3<0:
            relative_angle_gccPhat = 180

        ##########     When sound source is in bordeline of Quarter 2 and Quarter 3   ########
        elif T2==0 and T3<0:
            relative_angle_gccPhat = -90    

        #########      When sound source is in bordeline of Quarter 3 and Quarter 4   ########
        elif T1==0 and T3>0:
            relative_angle_gccPhat = 0

        else:
            print("There is some error in GCC-PHAT!")
            
        print("The DOA based on GCC-PHAT is: ", relative_angle_gccPhat)

        return relative_angle_gccPhat



      
            

