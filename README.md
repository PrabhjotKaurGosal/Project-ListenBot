# Project-ListenBot
This repo contains code for the ListenBot project
[Code to record, classify, localize sounds and then move Misty II towards the sound source.]

To run the code, it is preferable to first create an Anaconda environment and install various packages. The minimum packages to be isntalled are:
1. Packages related to Misty II Python SDK: https://pypi.org/project/Misty-SDK/. The detailed instructions and examples can be found at these links: https://prabhjotkaurgosal.com/getting-started-with-misty-ii-robot-and-its-python-sdk-page2/ and https://github.com/MistyCommunity/Python-SDK
2. Packagees related to sound classification using PANNs Inference. The instructions can be found here: https://github.com/qiuqiangkong/panns_inference

After the set up is complete, the code can be run from the main file. As an example, the conda environment created is called SoundClassification_Misty.
1. Navigate to the directroy where the main file and helper classes are present using the terminal
  - $cd Desktop
  - $cd SoundClassification
2. Activate the conda environment
  - $conda activate SoundClassification_Misty
  
Then, launch the Visual Studio Code. 



### References:
1. https://pypi.org/project/Misty-SDK/ - Link to Python SDK for Misty II
2. https://github.com/MistyCommunity/Python-SDK - Examples to run Python SDK for Misty II
3. https://prabhjotkaurgosal.com/getting-started-with-misty-ii-robot-and-its-python-sdk-page2/ - More detailed steps on installation and getting Misty II up and running using the Python SDK
4. https://github.com/qiuqiangkong/panns_inference - PANNs inference for sound classification
