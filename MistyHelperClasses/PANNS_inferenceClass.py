# PANNs - inference in a class
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels

class AudioClassificationClass(object):
    def __init__(self, audioFilePath):
        self.audioFilePath = audioFilePath

    def print_audio_tagging_result(self, clipwise_output):
        """Visualization of audio tagging result.
        Args:
        clipwise_output: (classes_num,)
        """
        sorted_indexes = np.argsort(clipwise_output)[::-1]
        Sounds_detected = []
        Sounds_detected_confidence = []

        # Print audio tagging top probabilities
        for k in range(1):
            print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], clipwise_output[sorted_indexes[k]]))
            d = (np.array(labels)[sorted_indexes[k]], clipwise_output[sorted_indexes[k]])
        Sounds_detected.append(d[0])
        Sounds_detected_confidence.append(d[1])
        max_confidence = Sounds_detected_confidence[0]
        max_confidence_Sound = Sounds_detected[0]
        #print("The sound type,", max_confidence_Sound, "is detected with confidence score of, ", max_confidence)

        return max_confidence_Sound, max_confidence


    def plot_sound_event_detection_result(self, framewise_output):
        """Visualization of sound event detection result. 
        Args:
        framewise_output: (time_steps, classes_num)
        """
        out_fig_path = 'results/sed_result.png'
        os.makedirs(os.path.dirname(out_fig_path), exist_ok=True)

        classwise_output = np.max(framewise_output, axis=0) # (classes_num,)

        idxes = np.argsort(classwise_output)[::-1]
        idxes = idxes[0:5]

        ix_to_lb = {i : label for i, label in enumerate(labels)}
        lines = []
        for idx in idxes:
            line, = plt.plot(framewise_output[:, idx], label=ix_to_lb[idx])
            lines.append(line)

        plt.legend(handles=lines)
        plt.xlabel('Frames')
        plt.ylabel('Probability')
        plt.ylim(0, 1.)
        plt.savefig(out_fig_path)
        print('Save fig to {}'.format(out_fig_path))

#if __name__ == '__main__':
    def classifyAudio_main(self):
        """Example of using panns_inferece for audio tagging and sound evetn detection.
        """
        device = 'cpu' # 'cuda' | 'cpu'
        self.audio_path = self.audioFilePath
        (audio, _) = librosa.core.load(self.audio_path, sr=32000, mono=True)
        audio = audio[None, :]  # (batch_size, segment_samples)

        print('------ Audio tagging ------')
        at = AudioTagging(checkpoint_path=None, device=device)
        (clipwise_output, embedding) = at.inference(audio)
        """clipwise_output: (batch_size, classes_num), embedding: (batch_size, embedding_size)"""

        sound_type, confidence_score =  self.print_audio_tagging_result(clipwise_output[0])
        
        # Uncomment the following code if sound event detection is needed
        # print('------ Sound event detection ------')
        # sed = SoundEventDetection(checkpoint_path=None, device=device)
        # framewise_output = sed.inference(audio)
        # """(batch_size, time_steps, classes_num)"""

        # self.plot_sound_event_detection_result(framewise_output[0])

        return sound_type, confidence_score