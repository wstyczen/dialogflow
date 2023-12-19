#!/usr/bin/env python3.6
import rospy

from speech_recognition import RequestError, UnknownValueError

from fallback_action_runner import FallbackActionRunner, FallbackAction
from human_interactions.msg import (
    TurnToHumanGoal,
)
from human_interactions.clients.turn_to_human_action_client import (
    TurnToHumanActionClient,
)
from sound_processing.enhance_audio import AudioEnhancement
from speech_to_text import speech_to_text
from text_to_speech import play_tts
from vad import VAD


class VoiceCommunication:
    """
    Class for managing voice communication scenarios with the robot.

    Attributes:
        _vad (VAD): VAD instance for voice activity detection.
        _turn_to_human_client (TurnToHumanActionClient): Client for the TurnToHuman action server.
        _fallback_action_runner (FallbackActionRunner): Fallback action runner for handling various scenarios.
    """

    def __init__(self):
        """
        Initialize a VoiceCommunication instance.
        """
        self._vad = VAD()
        self._vad.open_audio_stream()

        # Client for the TurnToHuman action server.
        self._turn_to_human_client = TurnToHumanActionClient()

        # Fallback action runner.
        self._fallback_action_runner = FallbackActionRunner()

    def run_fallback_action(self, action, *args):
        """
        Run a fallback action. If the fallback action limit for how many times
        in a row it can be perfiormed has been reached, the system will instead
        shut down.

        Args:
            action (FallbackAction): The action to perform.
        """
        if self._fallback_action_runner.is_action_count_limit_reached(action):
            print(
                f"Action limit for '{action.value}' has been reached.",
                f"Performed {self._fallback_action_runner.get_times_performed(action)} time(s) already.",
            )
            self._fallback_action_runner.run(FallbackAction.NOTIFY_SUPPORT)
            self.shut_down()

        return self._fallback_action_runner.run(action, *args)

    def on_stt_fail(self):
        """
        Default action to run when speech-to-text fails outright or the response
        probability is really low.
        """
        self.run_fallback_action(FallbackAction.MOVE_CLOSER_TO_SPEAKER)

    def shut_down(self):
        play_tts("Shutting down.")
        # TODO: Return the robot to the default pose ??
        exit()

    def confirm_receiving_command(self, command_transcription):
        play_tts(f"Voice command '{command_transcription}' received.")

    def run(self):
        """
        Initiates the voice communication system for a robot.

        This method always performs the following steps:
            1. Wait for wake word (keyword detection loop).
            2. When detected, move the robot to face the speaker.
            3. Notify the person of the readiness to take commands.
            4. Record a voice command.
            5. Enhance the audio.
            6. Use speech-to-text API to convert the command to text.
            7. Pass the command to a intent detection system (UNIMPLEMENTED).

        There are various ways the scenario above can fail. Fallback mechanisms
        are in place to handle such cases (FallbackAction, FallbackActionRunner).

        Instant failure conditions:
            1. Failure to initialize VAD module (necessary to record & analyze
                the audio).
            2. Failure to send an API request (necessary to parse the audio).
                May be caused by no internet connection.
        """
        # Wait for keyword.
        print("Running keyword detection.")
        if not self._vad.run_wake_word_detection():
            print("Failed to run keyword detection. Aborting.")
            return 1

        # When keyword is detected rotate the robot to face the human.
        print("Facing the human.")
        self._turn_to_human_client.send_goal(TurnToHumanGoal())
        # TODO: Handle failure of the request here ??
        self._turn_to_human_client.wait_for_result()

        # Notify the person of the readiness to take commands.
        play_tts("How can I help you?")

        while True:
            # Record the voice command.
            print("Recording voice command.")
            audio_path = self._vad.record_voice_command()

            # Calculate average volume of the audio (dB).
            # Needs to be done before audio enhancement (normalization).
            average_volume = AudioEnhancement.get_average_volume(audio_path)
            print(f"Average audio volume: {average_volume:.2f} dB.")
            # If the volume is really low, ask the speaker to repeat their
            # command louder.
            if average_volume < -40:
                self.run_fallback_action(FallbackAction.REQUEST_HIGHER_VOLUME)
                continue

            # Enhance recorded audio.
            print("Enhancing recorded audio quality.")
            AudioEnhancement(audio_path).enhance()

            print("Interpreting the voice command.")
            # Interpret the command (speech-to-text).
            # - If the stt fails or the response certainty is really low,
            # run the default emergency action.
            # - If response certainty is decent, but still uncertain ask for
            # confirmation.
            # - If the certainty is really high, assume the response is correct.
            STT_VALID_PROBABILITY_THRESHOLD = 0.9
            STT_MAYBE_PROBABILITY_THRESHOLD = 0.75
            stt_response = speech_to_text(audio_path)
            if not stt_response.error:
                text = stt_response.transcript
                confidence = stt_response.confidence
                print(
                    f"Interpreted the voice command as: '{text}'  with probability of {confidence:.2f}."
                )
                if confidence < STT_MAYBE_PROBABILITY_THRESHOLD:
                    print(
                        f"Speech-to-text probability too low (<{STT_MAYBE_PROBABILITY_THRESHOLD})."
                    )
                    self.on_stt_fail()
                    continue
                elif confidence < STT_VALID_PROBABILITY_THRESHOLD:
                    print(
                        f"Speech-to-text probability uncertain (<{STT_VALID_PROBABILITY_THRESHOLD})."
                    )
                    # Ask the user for confirmation that the transcription of
                    # the command is correct.
                    # If the speaker gives affirmation proceed with the
                    # transcription, otherwise run another fallback action.
                    if not self.run_fallback_action(
                        FallbackAction.ASK_FOR_CONFIRMATION, text
                    ):
                        continue

                # If reached here that means that the audio has been transcribed
                # with an acceptable confidence level, high enough to proceed.
                self.confirm_receiving_command(stt_response.transcript)

            elif stt_response.error == RequestError:
                return 2

            elif stt_response.error == FileNotFoundError:
                return 3

            elif stt_response.error == UnknownValueError:
                print(f"Speech-to-text could not produce a result for the audio.")
                self.on_stt_fail()
                continue

            # Detecting intent & acting accordingly.
            print("Passing the text command for further interpretation.")
            # TODO: Replace with a call to intent detection when a stable
            # working version is available.
            print("Intent detection / actions not available.")
            # intent_detection_sucessful = False

            # If intent detection is not successful, ask the person to rephrase
            # their last command.
            # if not intent_detection_sucessful:
            #     print("Intent detection failed.")
            #     self.run_fallback_action(FallbackAction.REQUEST_REPHRASING,
            #                                 stt_response.transcript)
            #     continue

            # If reached here then the command was successfully processed / executed.
            return 0


if __name__ == "__main__":
    rospy.init_node("voice_communication", anonymous=True)

    VoiceCommunication().run()
