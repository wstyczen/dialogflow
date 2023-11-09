#!/usr/bin/env python3.6
# encoding: utf8

import rospy
from human_interactions.msg import (
    TurnToHumanGoal,
    MoveToHumanGoal,
)
from human_interactions.clients.turn_to_human_action_client import (
    TurnToHumanActionClient,
)
from human_interactions.client.move_to_human_action_client import (
    MoveToHumanActionClient,
)
from sound_processing.enhance_audio import AudioEnhancement
from std_msgs.msg import String

from speech_recognition import RequestError, UnknownValueError

from vad import VoiceActivationDetector
from text_to_speech import play_tts
from speech_to_text import speech_to_text


class ExtendedVAD:
    def __init__(self):
        self._vad = VoiceActivationDetector()

        # Client for the TurnToHuman action server.
        self._turn_to_human_client = TurnToHumanActionClient()

        # Client for MoveToHuman action server.
        self._move_to_human_client = MoveToHumanActionClient()

        # Subscriber to monitor new recordings saved from VAD.
        self._recording_saved_subscriber = rospy.Subscriber(
            rospy.get_param("audio_file_topic"), String, self._on_command_recorded
        )

        # Index of the vad step to be used in logs.
        self._step_index = 0

    def _on_command_recorded(self, file_path):
        self._audio_path = file_path.data

    def print_step(self, msg):
        self._step_index = self._step_index + 1
        print(f"{self._step_index}. {msg}")

    def run_emergency_action(self):
        print("Running emergency action.")

        # Notify the person of the action.
        print("Playing the audio notification of the action.")
        play_tts("Command not recognized. Moving closer for better audio quality.")

        # Move closer to the person.
        print("Moving near the human.")
        self._move_to_human_client.send_goal(MoveToHumanGoal())
        self._move_to_human_client.wait_for_result()

        # Ask the person to repeat the command.
        print("Asking the person to repeat the command.")
        play_tts("Please repeat previous command.")

    def run(self):
        """
        Runs an extended VAD scenario.
        """
        # Open the audio stream for the vad.
        self._vad.open_audio_stream()

        # Keyword detection.
        self.print_step("Running keyword detection.")
        if not self._vad.run_wake_word_detection():
            print("Failed to run keyword detection. Aborting.")
            return 1

        # When keyword is detected rotate the robot to face the human.
        self.print_step("Facing the human.")
        self._turn_to_human_client.send_goal(TurnToHumanGoal())
        self._turn_to_human_client.wait_for_result()

        # When the robot is facing the human it should say that it is
        # ready for a command.
        self.print_step("Asking for a voice command.")
        play_tts("I am ready for a voice command.")

        while True:
            # Record the voice command.
            self.print_step("Recording the voice command.")
            audio_path = self._vad.record_voice_command()

            # Enhance recording audio quality.
            self.print_step("Enhancing recorded audio quality.")
            AudioEnhancement(audio_path).enhance()

            # Interpret the command (speech-to-text).
            self.print_step("Interpreting the voice command.")
            should_run_emergency_action = False
            STT_PROBABILITY_THRESHOLD = 0.9
            try:
                stt_text, stt_probability = speech_to_text(audio_path)
                print(
                    f"Interpreted voice command as: : '{stt_text}'  with probability of {stt_probability:.2f}."
                )
                if stt_probability < STT_PROBABILITY_THRESHOLD:
                    print(f"Speech-to-text probability too low.")
                    should_run_emergency_action = True
            except RequestError as e:
                print(f"Speech-to-text API request failed: '{e}'.\nQuitting_scenario.")
                return 2
            except UnknownValueError:
                print(f"Speech-to-text could not produce a result.")
                should_run_emergency_action = True

            # If stt was not successful or the probability is too low,
            # run the emergency action in order to get better audio.
            if should_run_emergency_action:
                self.run_emergency_action()
                # After performing the emergency action, move_back to the step
                # of recording voice command.
                continue

            # Detecting intent & acting accordingly.
            # Not used since no working version is available.
            # ...
            print("Intent detection...")

            return 0


if __name__ == "__main__":
    rospy.init_node("extended_vad", anonymous=True)

    ExtendedVAD().run()
