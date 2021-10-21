from mistyPy.Robot import Robot
from mistyPy.Events import Events
import matplotlib.pyplot as plt 
import numpy as np
import base64
import requests
import json
import time
import wave
from azure.cognitiveservices.speech import AudioDataStream, SpeechConfig, SpeechSynthesizer, SpeechSynthesisOutputFormat
from azure.cognitiveservices.speech.audio import AudioOutputConfig 

def ttsText(ipAddress):

    speech_key, service_region = "64ee2c3ca69847ce8d25f741d7f5a26b", "eastus"
    speech_config = SpeechConfig(subscription=speech_key, region=service_region)

    audio_config = AudioOutputConfig(filename="file.wav")

    # Creates a speech synthesizer using the default speaker as audio output.
    speech_synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)


    #print("Type some text that you want to speak...")
    #text = input()
    ssml_string = open("items2.xml", "r").read()
    speech_synthesizer.speak_ssml_async(ssml_string)

    enc = base64.b64encode(open("file.wav", "rb").read())
    eenc = enc.decode("utf-8")

    parameters = {"FileName": "file.wav", "Data": eenc, "ImmediatelyApply": True, "OverwriteExisting": True}
    #text = json.dumps(parameters, separators=(',', ':')) #Convert to JSON (optional)
    url = "http://" + ipAddress + "/api/audio"
    Qr1 = requests.post(url, json=parameters) # Note, POST request for Misty expects a JSON payload
    print(Qr1)

def changeLight(red, green, blue):
    parameters = {"red": red, "green": green, "blue": blue}
    text = json.dumps(parameters, separators=(',', ':')) #Convert to JSON (optional)
    url = "http://" + ipAddress + "/api/led"
    Qr1 = requests.post(url, json=parameters) # Note, POST request for Misty expects a JSON payload
    print(Qr1.status_code)

def writexml(text):
    # create the file structure
    data = ET.Element('speak')
    data.set("version", "1.0")
    data.set("xmlns", "http://www.w3.org/2001/10/synthesis")
    data.set("xml:lang", "en-US")
    items = ET.SubElement(data, 'voice')
    item1 = ET.SubElement(items, 'prosody')
    item1.set('volume','100')
    items.set('name',"en-US-AriaNeural")
    item1.text = text

    # create a new XML file with the results
    mydata = ET.tostring(data, encoding="unicode", method='xml')
    myfile = open("items2.xml", "w")
    myfile.write(mydata)

writexml("Hello")
ttstext(ipAddress)
