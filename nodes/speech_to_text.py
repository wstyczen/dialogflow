#!/usr/bin/env python3.6
import argparse
import os

import rospy
from rospkg import RosPack
import speech_recognition as sr

PACKAGE_PATH = RosPack().get_path("dialogflow")
AUDIO_SAMPLES_DIR = os.path.join(PACKAGE_PATH, "audio_samples")


def speech_to_text(
    audio_path
):
    """
    Interprets audio from a file using speech-to-text model and
    SpeechRecognition library and returns a string containing the infered text.

    Args:
        audio_path (str): Path to the audio file to be interpreted.
    """
    speech_recognizer = sr.Recognizer()

    with sr.AudioFile(audio_path) as f:
        audio_data = speech_recognizer.record(f)

    try:
        # APIs for a lot of different engines are available.
        text = speech_recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Audio recognition failed."
    except sr.RequestError as e:
        return f"Request failed: {e}"

if __name__ == "__main__":
    rospy.init_node("speech_to_text")

    print("Performing speech to text:")
    for file_path in rospy.get_param("~audio_files").split():
        print(f"- '{file_path}' --> '{speech_to_text(file_path)}'")
