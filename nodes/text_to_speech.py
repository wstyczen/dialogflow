#!/usr/bin/env python3.6
import os

import rospy
from rospkg import RosPack

import pygame
from gtts import gTTS

PACKAGE_PATH = RosPack().get_path("dialogflow")
DEFAULT_TTS_PATH = os.path.join(PACKAGE_PATH, "tts_out.mp3")


def text_to_speech(text, output_file=DEFAULT_TTS_PATH):
    """
    Use text-to-speech to generate audio and save to a file.

    Args:
        text (str): Text to convert to speech.
        output_file (str): Path to save the generated audio file in.
    """
    try:
        tts = gTTS(text)
        tts.save(output_file)
        # print(f"Saved tts of '{text}' to '{output_file}'.")
    except Exception as e:
        print(f"Failed to generate tts.\nError: {e}")


def play_audio(file_path):
    """
    Play the sound from audio file in the background.

    Args:
        file_path (str): Path to the audio file.
    """
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)

    # Play the audio file in the background.
    pygame.mixer.music.play()

    # Wait for the audio to finish playing.
    while pygame.mixer.music.get_busy():
        pass


def play_tts(text):
    """
    Plays the text-to-speech of given string in the background.

    Args:
        text (str): Text to play in the background.
    """
    text_to_speech(text, DEFAULT_TTS_PATH)
    play_audio(DEFAULT_TTS_PATH)


if __name__ == "__main__":
    rospy.init_node("text_to_speech")

    play_tts(rospy.get_param("~text"))
