#!/usr/bin/env python3.6
import rospy
from rospkg import RosPack

import speech_recognition as sr

PACKAGE_PATH = RosPack().get_path("dialogflow")


def speech_to_text(audio_path):
    """
    Interprets audio from a file using speech-to-text model and
    SpeechRecognition library and returns a string containing the infered text.

    Args:
        audio_path (str): Path to the audio file to be interpreted.

    Returns:
        response (tuple[str, float]): Return the most likely transcript and its
            probability.
    """
    speech_recognizer = sr.Recognizer()

    with sr.AudioFile(audio_path) as audio:
        # Adjust for noise in the audio beforehand, gives mixed results so far.
        # speech_recognizer.adjust_for_ambient_noise(source=audio, duration=0.5)

        audio_data = speech_recognizer.record(audio)

    # APIs for a lot of different engines are available, ie recognize_bing()
    response = speech_recognizer.recognize_google(
        audio_data, language="en-US", show_all=True
    )
    # Get the prediction with highest probability from response.
    prediction = response["alternative"][0]
    return prediction["transcript"], prediction["confidence"]


if __name__ == "__main__":
    rospy.init_node("speech_to_text")

    print("Performing speech to text:")
    for file_path in rospy.get_param("~audio_files").split():
        print(f"- '{file_path}' --> ", end="")
        try:
            transcription, probability = speech_to_text(file_path)
            print(f"transcription: '{transcription}', probability: {probability:.2f}")
        except:
            print(f"FAILED")
