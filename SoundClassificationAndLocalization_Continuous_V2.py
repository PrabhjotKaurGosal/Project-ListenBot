#############################  Main file to CLASSIFY sounds and MOVE Misty to the SOUND source ##################
# In this version of the file, the sound is recorded CONTINUOUSLY, processed by the GCC-PHAT to find the time delay (TDOA) between different channels and an
# azimuth angle to the sound source is calculated. Further, Misty moves (orients) towards the estimated direction of the sound source (estimated azimuth angle)
# In addition to calculating the azimuth angle for the sound source, sound is also classified by PANNs based neural network

from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.EventFilters import EventFilters
from MistyHelperClasses.RegisterEventsClass import RegisterMistyEvents
from MistyHelperClasses.PANNS_inferenceClass import AudioClassificationClass
from MistyHelperClasses.AudioAnalysisClass import AudioAnalysisClass
import json
import requests
from time import sleep
import base64
import wave
import librosa as lr
import os

# POST request: Start recording
def RecordAudio(ip_address, file_to_record):
    file = json.dumps(file_to_record, separators=(',', ':'))   #file_to_record = {"FileName": filename}, where filename = "ABC.wav"
    url = "http://" + ip_address + "/api/audio/raw/record/start"
    r3_A = requests.post(url, json=file)
    return (r3_A.status_code)

# POST request: Stop recording
def StopRecordAudio(ip_address):
    url = "http://" + ip_address + "/api/audio/record/stop"
    r3_B = requests.post(url)
    return (r3_B.status_code)

#Convert base64 string to .wav file
def ConvertAudio_base64stringToWav(base64string_data, file_to_record):
    encoded_audio  = list(base64string_data.values())[0]
    encoded_audio_value = list(encoded_audio.values())[0]
    #wav_file = open("temp.wav", "wb")
    wav_file = open(filename, "wb")
    decode_string = base64.b64decode(encoded_audio_value)
    wav_file.write(decode_string)

if __name__ =="__main__":
    #try:
            # First create the robot object
    ip_address = "IP ADDRESS"
    misty = Robot(ip_address)
    Dist_12 = 0.094
    Dist_13 = 0.082
    Dist_23 = 0.046
    while(1):

     #########################  START recording then REGISTER for audio event, wait for T seconds and the STOP recording #################
        # Start recording audio
        file_to_record = {"FileName": "AUGUST_1_raw.wav"}
        recording_status = RecordAudio(ip_address, file_to_record)
        print("The status of the Start Recording Audio request is: ", recording_status)

    #  wait for T seconds
        T = 0.5
        sleep(T)  

    #  Stop Recording audio
        stop_recordAudio_status = StopRecordAudio(ip_address)
        print("The status of the Stop Recording Audio request is: ",stop_recordAudio_status)

    #  Get the recorded file from Misty
        url = "http://" + ip_address + "/api/audio?FileName=AUGUST_1_raw.wav&Base64=True"  # NEED to revisit this code to prevent overriting the previous file
        r5 = requests.get(url, headers={'Content-Type': 'application/json'})
        print("The status of the GET FILE from Misty is: ", r5.status_code)

    #  Convert the file returned back from Misty to .wav format
        audioData_base64string = r5.json()
        filename = file_to_record["FileName"]
        ConvertAudio_base64stringToWav(audioData_base64string, filename)  # convert base64 string to .wav format

    ######################## CLASSIFY SOUND using ML (panns-inference) #######################
    #  Classify sound using PANNs neural nets
        directorySound = os.fsencode("/home/prabhjot/Desktop/SoundClassification")
        audio_path = (os.path.join(directorySound.decode("utf-8"), filename))
        SoundClassification_ = AudioClassificationClass(audio_path)
        SoundClassification_.classifyAudio_main()  ## NEED to revisit this code to have the class/function resturn the result in an array/list/dictionary

    #  Check if the classified sound is relevant 
        RelevantDomesticSounds = [] # Add - TO BE COMPLETED
        
    ####################### GET THE DIRECTION of SOUND #####################
        SoundLocalization = AudioAnalysisClass(audio_path, Dist_12, Dist_13, Dist_23) # Create an instance of the class AudioAnalysisClass -
    
    #  Extract audio channels from the raw audio
        wav = wave.open(audio_path)
        channel_idx = ["ch1.wav", "ch2.wav", "ch3.wav"]
        for i in range(3):
            channel_path = (os.path.join(directorySound.decode("utf-8"), channel_idx[i]))
            SoundLocalization.save_wav_channel(channel_path, wav, i)
            
        #SoundLocalization.save_wav_channel('ch1_1.wav', wav, 0)
        #SoundLocalization.save_wav_channel('ch2_1.wav',wav, 1)
        #SoundLocalization.save_wav_channel('ch3_1.wav', wav, 2)

    #  Read the audio files  ##### CHANGE THE SAMPLING FREQUENCY
        ch1, sfreq = lr.load(os.path.join(directorySound.decode("utf-8"), channel_idx[0]), sr=44100)
        ch2, sfreq = lr.load(os.path.join(directorySound.decode("utf-8"), channel_idx[1]), sr=44100)
        ch3, sfreq = lr.load(os.path.join(directorySound.decode("utf-8"), channel_idx[2]), sr=44100)
        
        #ch1, sfreq = lr.load('ch1.wav') # sfreq is the sampling frequency at which the signal is sampled
        #ch2, sfreq = lr.load('ch2.wav')
        #ch3, sfreq = lr.load('ch3.wav')
  
    #  Get the direction of arrival (angle in degrees)
        DOA = SoundLocalization.CalculateDOA(ch1,ch2,ch3, sfreq)

    #####################  MOVE MISTY TOwWARDS THE SOUND #####################
    #  REGISTER to IMU and GET the YAW value 
        MistyLiveData = RegisterMistyEvents(misty, "IMU_DATA",Events.IMU, None) # Create an instance of the class RegisterMistyEvents
        IMU_Data  = MistyLiveData.RegisterForEvents_generic()
        IMU_yaw = IMU_Data['yaw']
        print("The IMU yaw is: ", IMU_yaw)

    #  Move Misty's body towards the sound source (given by the direction of arrival of sound)
        absolute_angle = IMU_yaw + DOA
        time = ((abs(DOA))/20)*1000
        print("The absolute_angle is: ", absolute_angle)

        misty.DriveArc(absolute_angle, 0, time, False) #Uncomment
        # misty.Stop();
        print("Misty moved!")
        
        ## Move FORWARD in a straight line and make it stop after 5 seconds

        #misty.KeepAlive()
       
    #except Exception as ex:
        #print(ex)
    #finally:
    # Unregister events if they aren't all unregistered due to an error
        #misty.UnregisterAllEvents()





#################################### UNUSED CODE################################################################################################

    # REGISTER to the ACTUATOR event and get the yaw for the head #
        # MistyLiveData = RegisterMistyEvents(misty, "Actuator_headYaw", Events.ActuatorPosition, [EventFilters.ActuatorPosition.HeadYaw])
        # Head_YawData = MistyLiveData.RegisterForEvents_generic()
        # Head_Yaw = Head_YawData ['value']
        # print("The Head yaw is: ", Head_Yaw)

    # REGISTER for event to get the direction of sound and other meta data #
        # MistyLiveData = RegisterMistyEvents(misty, "SoundTrack",Events.SourceTrackDataMessage, None)
        # Sound_Dta  = MistyLiveData.RegisterForEvents_generic()
        # dir_Speech= Sound_Dta['degreeOfArrivalSpeech']
        # dir_Noise = Sound_Dta['degreeOfArrivalNoise']
        # volume_360 = Sound_Dta['voiceActivityPolar']
        # print("The dir_Speech is: ", dir_Speech)
        # print("The dir_Noise is: ", dir_Noise)
        # print("The volume is: ", volume_360)
