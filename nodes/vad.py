#!/usr/bin/env python3.6
# encoding: utf8

#
# Copyright 2018 Picovoice Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import wave
import pyaudio
import struct
import datetime
import time
from threading import Thread
import pvporcupine
from struct import pack
from multiprocessing import Queue
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi
import collections
from array import array
from struct import pack
from scipy.signal import butter, lfilter, lfilter_zi

try:
    import rospy
    from std_msgs.msg import String, Bool
    from rospkg import RosPack
except:
    rospy.logerr(f"\033[91mFailed to access ROS modules.\nAborting.")
    exit()


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


class AudioChannel:
    LEFT = 0
    RIGHT = 1
    FILTERED_LEFT = 2
    FILTERED_RIGHT = 3


class VoiceActivationDetector(Thread):
    """
    Voice Activation Detector (VAD) - utilizes Porcupine library for wake word
    detection. Creates an input audio stream from microphone(s) and monitors
    it for occurences of the wake-word(s). Upon detection records a voice
    command and saves it to a file.
    """

    FRAME_RATE = 16000
    FRAME_LENGTH = 512
    AUDIO_CHANNELS = [
        AudioChannel.LEFT,
        AudioChannel.RIGHT,
        AudioChannel.FILTERED_LEFT,
        AudioChannel.FILTERED_RIGHT,
    ]

    def __init__(
        self,
        access_key,
        keyword_file_paths,
        sensitivities,
    ):
        """
        Constructor.

        :param keyword_file_paths: List of absolute paths to keyword files.
        :param sensitivities: Sensitivity parameter for each wake word. For more information refer to
        'include/pv_porcupine.h'. It uses the
        same sensitivity value for all keywords.
        """
        super(VoiceActivationDetector, self).__init__()
        self._access_key = access_key
        self._keyword_file_paths = keyword_file_paths
        self._sensitivities = sensitivities or 0.5

        self.recorded_frames = Queue()
        self._activate_vad_received = False

        # Audio stream.
        self._pyaudio_handle = pyaudio.PyAudio()
        self._audio_stream = None

        print("Connecting to publisher")
        self.pub = rospy.Publisher(
            rospy.get_param("audio_file_topic"), String, queue_size=10
        )

        # Used to skip wake-word detection step and go straight to recording the
        # command.
        # Can be triggered from cmd by:
        # rostopic pub /activate_vad std_msgs/Bool 1
        self.sub_activate_vad = rospy.Subscriber(
            "/activate_vad", Bool, self._activate_vad_callback
        )

        print("Initialization done.")

    def __del__(self):
        if self._pyaudio_handle is not None:
            self._pyaudio_handle.terminate()
        self.close_audio_stream()

    def _activate_vad_callback(self, _):
        print("Vad triggered remotely by topic.")
        self._activate_vad_received = True

    def _audio_stream_callback(self, in_data, frame_count, time_info, status):
        def get_audio_split_to_channels(in_data):
            def process_channel(block, low_pass=400, high_pass=4000):
                """
                Returns the original and the filtered audio packed into chunks.
                """
                in_data = pack("<" + ("h" * len(block)), *block)

                b, a = butter_bandpass(
                    low_pass, high_pass, VoiceActivationDetector.FRAME_RATE, order=5
                )
                zi_global = lfilter_zi(b, a)

                filtered_block, zi = lfilter(b, a, block, zi=zi_global)
                filtered_block = filtered_block.astype(np.int16)
                chunk_to_analyze = pack(
                    "<" + ("h" * len(filtered_block)), *filtered_block
                )
                return (in_data, chunk_to_analyze)

            decoded_block = np.fromstring(in_data, "Int16")

            left_channel = decoded_block[0::2]
            right_channel = decoded_block[1::2]

            (left, filtered_left) = process_channel(left_channel)
            (right, filtered_right) = process_channel(right_channel)

            return left, filtered_left, right, filtered_right

        left, filtered_left, right, filtered_right = get_audio_split_to_channels(
            in_data
        )
        self.recorded_frames.put(
            {
                AudioChannel.LEFT: left,
                AudioChannel.FILTERED_LEFT: filtered_left,
                AudioChannel.RIGHT: right,
                AudioChannel.FILTERED_RIGHT: filtered_right,
            }
        )

        return "", pyaudio.paContinue

    def open_audio_stream(self):
        self._audio_stream = self._pyaudio_handle.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=VoiceActivationDetector.FRAME_RATE,
            input=True,
            frames_per_buffer=VoiceActivationDetector.FRAME_LENGTH,
            stream_callback=self._audio_stream_callback,
        )

    def close_audio_stream(self):
        if self._audio_stream is not None:
            self._audio_stream.close()

    def run_wake_word_detection(self):
        """
        Creates an input audio stream, initializes wake word detection (Porcupine) object, and monitors the audio
        stream for occurrences of the wake word(s).
        """
        # Porcupine engines for keyword detection.
        porcupine_engines = {}

        def print_keywords():
            num_keywords = len(self._keyword_file_paths)
            assert (
                num_keywords > 0
            ), "At least one keyword must be provided for detection."
            assert num_keywords == len(
                self._sensitivities
            ), "Please provide a single sensitivity for each keyword."

            print("Listening for keywords:")
            for i, keyword_file_path in enumerate(self._keyword_file_paths):
                keyword_name = (
                    os.path.basename(keyword_file_path)
                    .replace(".ppn", "")
                    .replace("_compressed", "")
                    .split("_")[0]
                )
                print(
                    "- '{}' (with sensitivity: {})".format(
                        " ".join(keyword_name.split("-")), self._sensitivities[i]
                    )
                )

        def initialize_porcupine_engines():
            """
            Create wake-word porcupine engines for each channel - one that checks
            original audio and one that checks filtered audio.
            """
            porcupine_engines.clear()
            for channel in self.AUDIO_CHANNELS:
                porcupine_engines[channel] = pvporcupine.create(
                    access_key=self._access_key, keyword_paths=self._keyword_file_paths
                )

        def shutdown_porcupine_engines():
            for engine in porcupine_engines.values():
                engine.delete()

        def was_keyword_detected(frame):
            """
            Return whether any of the wake-words were detected in the given
            frame.
            """
            # Porcupine's 'process' method checks the audio for wake words.
            # It returns -1 if no wake-word was detected.
            # If a wake-word was detected it returns its index.
            results = []
            for channel, engine in porcupine_engines.items():
                pcm = frame[channel]
                pcm = struct.unpack_from("h" * engine.frame_length, pcm)
                results.append(engine.process(pcm))

            return max(results) >= 0

        print_keywords()
        try:
            initialize_porcupine_engines()

            last_keyword_detection_time = datetime.datetime.now()

            def was_keyword_detected_within_timeout(time_out=1):
                time_diff = datetime.datetime.now() - last_keyword_detection_time
                return time_diff.seconds < time_out

            # Keyword detection loop.
            while True:
                try:
                    frame = self.recorded_frames.get(block=False)
                except:
                    continue

                if (
                    was_keyword_detected(frame)
                    and not was_keyword_detected_within_timeout()
                ) or self._activate_vad_received:
                    print("Keyword detected.")

                    # Clear manual activation.
                    self._activate_vad_received = False

                    # Shutdown the porcupine engines.
                    shutdown_porcupine_engines()

                    return True

        except:
            return False

    class VoicedFramesTracker:
        """
        Keeps track of the number of voiced frames within the given window.
        """

        # Minimum power for the frame to be considered 'voiced'.
        VOICED_POWER_THRESHOLD = 120

        def __init__(self, buffer_size):
            # 1 - voiced, 0 - silent
            self._buffer = collections.deque([0] * buffer_size, maxlen=buffer_size)

        def update(self, is_voiced):
            self._buffer.append(is_voiced)

        def get_num_voiced(self):
            return self._buffer.count(1)

        def get_num_silent(self):
            return self._buffer.count(0)

        @classmethod
        def is_voiced(cls, frame):
            power = np.mean(np.abs(frame))
            return int(power > cls.VOICED_POWER_THRESHOLD)

    def clear_recorded_frames(self):
        while True:
            try:
                self.recorded_frames.get(block=False)
            except:
                break

    def record_voice_command(self):
        # Consider command started after that many voiced frames.
        VOICED_FRAMES_THR = 10
        # After the command was started and that many silent frames passed, stop
        # recording early.
        SILENT_FRAMES_THR = 15
        # Time limit for recording in seconds.
        RECORDING_TIME_LIMIT = 5
        # Whether the recorded audio should be normalized before saving.
        DO_NORMALIZE = True

        # Window of frames observed constants.
        CHUNK_DURATION_MS = 30  # Supports 10, 20 and 30 (ms)
        WINDOW_DURATION = 500  # ms
        # Nr of frames, ie 500 ms / 30 ms ~= 16 frames
        WINDOW_LENGTH = int(WINDOW_DURATION / CHUNK_DURATION_MS)
        assert WINDOW_LENGTH >= max(
            VOICED_FRAMES_THR, SILENT_FRAMES_THR
        ), "The monitored window of frames can't be shorter then set thresholds."

        # Whether enough voiced frames where 'voiced'.
        got_voiced_frames = False
        # Whether enough frames where 'voiced' and latest frames were silent.
        got_a_sentence = False

        # Nr of frames to ignore at the start of command.
        ignore = 5

        # Initialize voiced frames trackers.
        voiced_frames_tracker_left = self.VoicedFramesTracker(buffer_size=WINDOW_LENGTH)
        voiced_frames_tracker_right = self.VoicedFramesTracker(
            buffer_size=WINDOW_LENGTH
        )

        # Clear any queued frames.
        self.clear_recorded_frames()

        # Initialize recording.
        raw_data_left = array("h")
        raw_data_right = array("h")
        StartTime = time.time()
        TimeUse = 0

        frames_str = ""

        def print_frames(is_frame_voiced):
            nonlocal frames_str
            VOICED_FRAME_REPR = "#"
            SILENT_FRAME_REPR = "-"

            frames_str += VOICED_FRAME_REPR if is_frame_voiced else SILENT_FRAME_REPR
            # Limit the length of displayed string.
            LINE_LENGTH_LIMIT = 100
            if len(frames_str) > LINE_LENGTH_LIMIT:
                frames_str = frames_str[len(frames_str) - LINE_LENGTH_LIMIT :]

            print(f"\r{frames_str}", end="\r")

        print("Recording:")
        # Record sound while the loop is active.
        while not got_a_sentence and TimeUse <= RECORDING_TIME_LIMIT:
            # Get the active audio frame.
            try:
                frame = self.recorded_frames.get(block=False)
            except Exception:
                # print("No frame available.")
                continue

            # Skip a few frames at the start, to make sure the 'wake-word' is
            # not included in the recording.
            if ignore > 0:
                ignore -= 1
                continue

            # Process audio frame.
            chunk_left = frame[AudioChannel.LEFT]
            chunk_right = frame[AudioChannel.RIGHT]
            filtered_left = frame[AudioChannel.FILTERED_LEFT][0:960]
            filtered_right = frame[AudioChannel.FILTERED_RIGHT][0:960]

            decoded_block_left = np.fromstring(filtered_left, "Int16")
            decoded_block_right = np.fromstring(filtered_right, "Int16")

            # Keep data for saving later.
            raw_data_left.extend(array("h", chunk_left))
            raw_data_right.extend(array("h", chunk_right))

            # Check if frame is 'voiced'.
            is_voiced_left = voiced_frames_tracker_left.is_voiced(decoded_block_left)
            is_voiced_right = voiced_frames_tracker_right.is_voiced(decoded_block_right)
            # Track the number of voiced frames.
            voiced_frames_tracker_left.update(is_voiced_left)
            voiced_frames_tracker_right.update(is_voiced_right)

            voiced_frames_left = voiced_frames_tracker_left.get_num_voiced()
            voiced_frames_right = voiced_frames_tracker_right.get_num_voiced()
            silent_frames_left = voiced_frames_tracker_left.get_num_silent()
            silent_frames_right = voiced_frames_tracker_left.get_num_silent()
            # print(
            #     f"Voiced: {np.mean([voiced_frames_left, voiced_frames_right])}, Silent: {np.mean([silent_frames_left, silent_frames_right])}",
            #     end="\r",
            # )

            # Display whether the frame is voiced.
            print_frames(is_voiced_left and is_voiced_right)

            # Note the start of command (enough 'voiced' frames).
            if (
                voiced_frames_left > VOICED_FRAMES_THR
                or voiced_frames_right > VOICED_FRAMES_THR
            ):
                got_voiced_frames = True
            # If the command has started (got enough 'voiced' frames) and enough
            # silent frames pass, end recording early.
            if (
                got_voiced_frames
                and silent_frames_left > SILENT_FRAMES_THR
                and silent_frames_right > SILENT_FRAMES_THR
            ):
                got_a_sentence = True

            # Update time recorded.
            TimeUse = time.time() - StartTime

        print(frames_str)
        if got_a_sentence:
            print("Ended early because of silent frames.")
        elif TimeUse > RECORDING_TIME_LIMIT:
            print("Time limit reached.")
        print("Recording ended.")

        # Normalize data before saving.
        def normalize(snd_data):
            "Average the volume out."
            MAXIMUM = 32767  # 16384
            times = float(MAXIMUM) / max(abs(i) for i in snd_data)
            r = array("h")
            for i in snd_data:
                r.append(int(i * times))
            return r

        if DO_NORMALIZE:
            raw_data_left = normalize(raw_data_left)
            raw_data_right = normalize(raw_data_right)

        # Save audio to a file.
        now = datetime.datetime.now()
        fname = now.strftime("/tmp/%m-%d-%Y-%H-%M-%S") + ".wav"

        def record_to_file(path, data, sample_width, rate):
            "Records from the microphone and outputs the resulting data to 'path'"
            (channel_left, channel_right) = data
            wf = wave.open(path, "wb")
            wf.setnchannels(2)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            for left, right in zip(channel_left, channel_right):
                left_frame = pack("<h", left)
                wf.writeframes(left_frame)
                right_frame = pack("<h", right)
                wf.writeframes(right_frame)
            wf.close()

        record_to_file(
            fname,
            (raw_data_left, raw_data_right),
            2,
            VoiceActivationDetector.FRAME_RATE,
        )
        print("Saved to recording to '%s'." % fname)

    def run(self):
        self.open_audio_stream()

        while True:
            if self.run_wake_word_detection():
                # Record voice command.
                self.record_voice_command()

                # Clear any queued frames.
                self.clear_recorded_frames()


if __name__ == "__main__":
    rospy.init_node("vad", anonymous=True)

    package_path = RosPack().get_path("dialogflow")
    HEY_RICO_KEYWORD_FILE_PATH = "Hey-Rico_en_linux_v2_1_0.ppn"
    VoiceActivationDetector(
        access_key="aDd541fQUB9+vb6KqcWV7kMBEvOkHQGV/bg7Z/1pbE1gcS0TmHzpYA==",
        keyword_file_paths=[os.path.join(package_path, HEY_RICO_KEYWORD_FILE_PATH)],
        sensitivities=[0.5],
    ).run()
