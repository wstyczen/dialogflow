#!/usr/bin/env python3.6
from enum import Enum

import actionlib
import rospy
from human_interactions.msg import (
    MoveToHumanAction,
    MoveToHumanGoal,
)

from speech_to_text import speech_to_text
from text_to_speech import play_tts
from vad import VAD


class FallbackAction(str, Enum):
    """Enum representing various fallback actions.

    Values:
        REQUEST_HIGHER_VOLUME: Ask the speaker to repeat the command at a higher volume.
        REQUEST_REPHRASING: Request the speaker to rephrase the last command.
        ASK_FOR_CONFIRMATION: Ask the user for confirmation to ensure the last voice command was understood correctly.
        MOVE_CLOSER_TO_SPEAKER: Move closer to the speaker and ask them to repeat the command.
        NOTIFY_SUPPORT: Notify support personnel before shutting down the voice communication. Used after attempts to communicate have failed continuously.
    """

    # USER FEEDBACK
    REQUEST_HIGHER_VOLUME = "Request higher volume"
    REQUEST_REPHRASING = "Request rephrasing"
    # USER CONFIRMATION
    ASK_FOR_CONFIRMATION = "Ask for confirmation"
    # FALLBACK MECHANISM
    MOVE_CLOSER_TO_SPEAKER = "Move closer to the speaker"
    # HUMAN INTERVENTION
    NOTIFY_SUPPORT = "Notify support"


class FallbackActionRunner:
    """
    This class executes a requested type of fallback action for the voice
    communication system.

    Attributes:
        ACTION_PERFORMANCE_LIMITS: Defines how many times each fallback action
            can be performed in a row before it should be aborted.
        _vad (VAD): VAD instance for recording user feedback.
    """

    ACTION_PERFORMANCE_LIMITS = {
        FallbackAction.REQUEST_HIGHER_VOLUME: 2,
        FallbackAction.REQUEST_REPHRASING: 2,
        FallbackAction.ASK_FOR_CONFIRMATION: 2,
        FallbackAction.MOVE_CLOSER_TO_SPEAKER: 1,
    }

    def __init__(self):
        """
        Initialize a FallbackActionRunner instance.
        """
        # VAD instance for recording user feedback.
        self._vad = VAD()

        # Client for the MoveToHuman action server.
        self._move_to_human_client = actionlib.SimpleActionClient(rospy.get_param("move_to_human_action_name"), MoveToHumanAction)
        self._move_to_human_client.wait_for_server()

        # Keep a history of performed actions.
        self._history = []

    def clear_history(self):
        self._history.clear()

    def get_times_performed(self, action):
        return self._history.count(action)

    def is_action_count_limit_reached(self, action):
        return self.get_times_performed(action) >= self.ACTION_PERFORMANCE_LIMITS.get(
            action, 1
        )

    def get_num_actions_performed(self):
        return len(self._history)

    def _request_higher_volume(self):
        """Ask the speaker to repeat the voice command at a higher volume."""
        play_tts("That was really quiet. Please repeat the command louder.")

    def _request_rephrasing(self, interpreted_command=None):
        """
        Request the user to rephrase their last command.

        Args:
            interpreted_command (str, optional): Last command's contents.
                If provided, it will be used when asking the user for confirmation.
        """
        if interpreted_command is None or len(interpreted_command) == 0:
            play_tts("I did not understand the last command.")
        else:
            play_tts(f"I do not understand what '{interpreted_command}' means.")

        play_tts("Please use different phrasing.")

    def _ask_for_confirmation(self, interpreted_command):
        """
        Ask the user for confirmation to ensure the last voice command received
        was parsed correctly.

        Args:
            interpreted_command (str): The interpreted command for confirmation.
        """
        assert interpreted_command is not None and len(interpreted_command) > 0
        play_tts(f"Just to make sure, was the last command '{interpreted_command}'?")

        class ResponseType(Enum):
            AFFIRMATIVE = "positive"
            NEGATIVE = "negative"
            UNDETERMINED = "undetermined"

        def process_response(response_text):
            cleaned_response = "".join(
                char.lower()
                for char in response_text
                if char.isalnum() or char.isspace()
            )

            # Check if the cleaned response matches the affirmative or negative
            # patterns.
            if any(
                s in cleaned_response
                for s in [
                    "no",
                    "nope",
                    "not",
                    "isn't",
                    "wasn't",
                    "never",
                    "negative",
                    "nah",
                    "nay",
                ]
            ):
                return ResponseType.NEGATIVE
            if any(
                s in cleaned_response
                for s in [
                    "yes",
                    "yeah",
                    "yep",
                    "sure",
                    "absolutely",
                    "of course",
                    "certainly",
                    "indeed",
                    "definitely",
                    "affirmative",
                    "ok",
                    "okay",
                    "fine",
                    "that is correct",
                    "that's correct",
                ]
            ):
                return ResponseType.AFFIRMATIVE
            return ResponseType.UNDETERMINED

        # Record user's response.
        self._vad.open_audio_stream()
        response_audio_path = self._vad.record_voice_command()
        self._vad.close_audio_stream()

        stt_response = speech_to_text(response_audio_path)
        response_type = process_response(stt_response.transcript)
        print(
            f"Intepreted response '{stt_response.transcript}' as: {response_type.value}"
        )
        if response_type == ResponseType.UNDETERMINED:
            return self._ask_for_confirmation(interpreted_command)
        elif response_type == ResponseType.AFFIRMATIVE:
            play_tts(
                "Thank you for the confirmation. I will proceed with this command."
            )
            print("Command interpretation confirmed.")
            return True
        else:
            play_tts("Okay, then please repeat your last command clearly.")
            print("Promping user to repeat their last command.")
            return False

    def _move_closer_to_speaker(self):
        """Move closer to the speaker and ask them to repeat the last command."""
        # Notify the person of the action.
        print("Playing the audio notification.")
        play_tts(
            "I couldn't hear your last command very well. I will move closer for better audio quality."
        )

        # Move closer to the person.
        print("Moving.")

        self._move_to_human_client.send_goal_and_wait(MoveToHumanGoal())

        # Ask the person to repeat the command.
        print("Asking the person to repeat the command.")
        play_tts("Please repeat the previous command.")

    def _notify_support(self):
        play_tts("Voice communication attempts have failed.")
        # TODO: Implement.
        pass

    def run(self, action, *args):
        """
        Run the requested type of fallback action and pass the rest argument to
        that method.

        Args:
            action (FallbackAction): Fallback action to be executed.
            *args: Additional arguments that may be required by the specific action method.
        """
        ACTIONS = {
            FallbackAction.REQUEST_HIGHER_VOLUME: self._request_higher_volume,
            FallbackAction.REQUEST_REPHRASING: self._request_rephrasing,
            FallbackAction.ASK_FOR_CONFIRMATION: self._ask_for_confirmation,
            FallbackAction.MOVE_CLOSER_TO_SPEAKER: self._move_closer_to_speaker,
            FallbackAction.NOTIFY_SUPPORT: self._notify_support,
        }

        self._history.append(action)

        action_method = ACTIONS.get(action)
        print(f"Running fallback action: '{action.value}'.")
        return action_method(*args)


if __name__ == "__main__":
    rospy.init_node("fallback_action_runner", anonymous=True)

    fallback_action_runner = FallbackActionRunner()
    fallback_action_runner.run(FallbackAction.MOVE_CLOSER_TO_SPEAKER)

    rospy.spin()
