#!/usr/bin/env python3.6
import json
from rospkg import RosPack
import os
from collections import namedtuple

import rospy
import speech_recognition as sr

from sound_processing.enhance_audio import AudioEnhancement


STTResponse = namedtuple("STTResponse", ["error", "transcript", "confidence"])


def speech_to_text(audio_path):
    """
    Interprets audio from a file using speech-to-text model and
    SpeechRecognition library and returns a string containing the infered text.

    Args:
        audio_path (str): Path to the audio file to be interpreted.

    Returns:
        response (STTResponse): Return the most likely transcript and its
            probability. If an error occurs it is returned instead.
    """
    Error = lambda ErrorType: STTResponse(ErrorType, "", 0)
    Prediction = lambda transcript, confidence: STTResponse(
        None, transcript, confidence
    )
    try:
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

        if "alternative" not in response:
            raise sr.UnknownValueError
        prediction = response["alternative"][0]
        rospy.loginfo(f"[STT] Successfully parsed '{audio_path}'.")
        return Prediction(
            transcript=prediction["transcript"], confidence=prediction["confidence"]
        )
    except FileNotFoundError:
        rospy.logerr(f"[STT] Audio file not found at '{audio_path}'.")
        return Error(FileNotFoundError)
    except sr.RequestError:
        rospy.logerr(f"[STT] Error sending request.")
        return Error(sr.RequestError)
    except sr.UnknownValueError:
        rospy.logerr(f"[STT] Could not parse the audio successfully.")
        return Error(sr.UnknownValueError)


if __name__ == "__main__":
    rospy.init_node("speech_to_text")

    engine_used = "sphinx"
    stt_results = {}

    print("Performing speech to text:")
    # Use the audio files specified in the launch file.
    for file_path in rospy.get_param("~audio_files").split():
        result = {}

        def save_result(response, key):
            if response.error:
                result[key] = {"error": str(response.error)}
            else:
                result[key] = {
                    "transcript": response.transcript,
                    "confidence": response.confidence,
                }

        # Try speech to text on normal audio.
        save_result(speech_to_text(file_path), "default_audio")

        # # Copy the file and enhance it.
        # TMP_AUDIO_PATH = "/tmp/tmp.wav"
        # os.system(f"cp {file_path} {TMP_AUDIO_PATH}")
        # AudioEnhancement(TMP_AUDIO_PATH).enhance(use_api=True)
        # # Try speech to text again on enhanced audio.
        # save_result(speech_to_text(TMP_AUDIO_PATH), "enhanced_audio")

        # stt_results[file_path] = result

    # Save results to a json file within a subdir of this package.
    results_dir_path = os.path.join(RosPack().get_path("dialogflow"), "stt_results")
    if not os.path.exists(results_dir_path):
        os.makedirs(results_dir_path)
    with open(
        os.path.join(results_dir_path, f"{engine_used}_results.json"), "w"
    ) as outfile:
        json.dump(stt_results, outfile)
