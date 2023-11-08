#!/usr/bin/env python3.6
# encoding: utf8

from enum import Enum

import rospy
from human_interactions.msg import (
    TurnToHumanGoal,
)
from human_interactions.clients.turn_to_human_action_client import (
    TurnToHumanActionClient,
)
from sound_processing.enhance_audio import AudioEnhancement
from std_msgs.msg import String

from vad import VoiceActivationDetector


class ExtendedVAD:
    def __init__(self):
        self._vad = VoiceActivationDetector()

        # Client for the TurnToHuman action server.
        self._turn_to_human_client = TurnToHumanActionClient()

        # Subscriber to monitor new recordings saved from VAD.
        self._recording_saved_subscriber = rospy.Subscriber(
            rospy.get_param("audio_file_topic"), String, self._on_command_recorded
        )
        self._recording_path = None

    def _on_command_recorded(self, file_path):
        self._recording_path = file_path.data

    def run(self):
        """
        Runs an extended VAD scenario.
        """
        # Open the audio stream for the vad.
        self._vad.open_audio_stream()

        while True:
            # 1. Keyword detection.
            print("1. Running keyword detection.")
            if not self._vad.run_wake_word_detection():
                print("Failed to run keyword detection. Aborting.")
                return -1

            # 2. When keyword is detected rotate the robot to face the human.
            print("2. Facing the human.")
            self._turn_to_human_client.send_goal(TurnToHumanGoal())
            self._turn_to_human_client.wait_for_result()

            # 3. When the robot is facing the human it should say that it is
            # ready for a command.
            print("3. [TODO] Asking for a voice command.")
            # TODO: Implement talker for Rico to speak.
            # ...

            # 4. Record the voice command.
            print("4. Recording the voice command.")
            recording_path = self._vad.record_voice_command()

            # 5. Enhance recording audio quality.
            print("5. Enhancing recorded audio quality.")
            AudioEnhancement(recording_path).enhance()

            # 6. Interpret the command (speech-to-text).
            print("6. Interpreting the voice command.")
            # TODO: Implement speech-to-text.
            # ...

            # 7. Detecting intent.
            # ...


if __name__ == "__main__":
    rospy.init_node("extended_vad", anonymous=True)

    ExtendedVAD().run()
