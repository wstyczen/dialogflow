#!/usr/bin/env python3.6
from enum import Enum

from human_interactions.clients.move_to_human_action_client import (
    MoveToHumanActionClient,
)
from human_interactions.msg import (
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
        REQUEST_THE_SPEAKER_TO_MOVE_CLOSER: Ask the person to move closer before initiating voice communication.
        NOTIFY_SUPPORT: Notify support personnel before shutting down the voice communication. Used after attempts to communicate have failed continuously.
    """

    # USER FEEDBACK
    REQUEST_HIGHER_VOLUME = "Requesting the person to repeat the command louder."
    REQUEST_REPHRASING = "Requesting the user to rephrase the command"
    # USER CONFIRMATION
    ASK_FOR_CONFIRMATION = "Asking the speaker for confirmation."
    # FALLBACK MECHANISM
    MOVE_CLOSER_TO_SPEAKER = "Moving closer to the speaker."
    REQUEST_THE_SPEAKER_TO_MOVE_CLOSER = "Requesting the speaker to move closer."
    # HUMAN INTERVENTION
    NOTIFY_SUPPORT = "Notifying support before shutting down the voice communication."


class FallbackActionRunner:
    """
    This class executes a requested type of fallback action for the voice
    communication system.

    Attributes:
        _move_to_human_client (MoveToHumanActionClient): Client for MoveToHuman action server.
        _vad (VAD): VAD instance for recording user feedback.
    """

    def __init__(self):
        """
        Initialize a FallbackActionRunner instance.
        """
        # Client for MoveToHuman action server.
        self._move_to_human_client = MoveToHumanActionClient()

        # VAD instance for recording user feedback.
        self._vad = VAD()

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
            if cleaned_response in [
                "no",
                "nope",
                "not",
                "isn't",
                "wasn't",
                "never",
                "negative",
                "nah",
                "nay",
            ]:
                return ResponseType.NEGATIVE

            if cleaned_response in [
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
            ]:
                return ResponseType.AFFIRMATIVE

            return ResponseType.UNDETERMINED

        # Record user's response.
        response_audio_path = self._vad.record_voice_command()
        try:
            response, _ = speech_to_text(response_audio_path)
            response_type = process_response(response)
            if response_type == ResponseType.UNDETERMINED:
                return self._ask_for_confirmation(interpreted_command)
            else:
                return True if response_type == ResponseType.AFFIRMATIVE else False
        except:
            # If stt failed, ask again.
            return self._ask_for_confirmation(interpreted_command)

    def _move_closer_to_speaker(self):
        """Move closer to the speaker and ask them to repeat the last command."""
        # Notify the person of the action.
        print("Playing the audio notification.")
        play_tts(
            "I couldn't hear your last command very well. I will move closer for better audio quality."
        )

        # Move closer to the person.
        print("Moving near the human.")
        self._move_to_human_client.send_goal(MoveToHumanGoal())
        self._move_to_human_client.wait_for_result()

        # Ask the person to repeat the command.
        print("Asking the person to repeat the command.")
        play_tts("Please repeat the previous command.")

    def _request_the_speaker_to_move_closer(self):
        """Ask the person to move closer before initiating voice communication."""
        play_tts("Please move closer before initiating voice communication.")

    def _notify_support(self):
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
            FallbackAction.REQUEST_THE_SPEAKER_TO_MOVE_CLOSER: self._request_the_speaker_to_move_closer,
            FallbackAction.NOTIFY_SUPPORT: self._notify_support,
        }

        action_method = ACTIONS.get(action)
        print(f"Running fallback action.\n{action.value}.")
        action_method(*args)
