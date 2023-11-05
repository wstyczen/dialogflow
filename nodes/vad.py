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
import sys

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

# packages for VAD
import collections
from array import array
from struct import pack
from scipy.signal import butter, lfilter, lfilter_zi

# from multiprocessing.queues import Queue
import itertools

has_ros = True
try:
    import rospy
    import actionlib
    from std_msgs.msg import String, Bool
    from rospkg import RosPack
    from dialogflow_actions.msg import (
        TurnToHumanGoal,
    )
    from dialogflow_actions.clients.turn_to_human_action_client import (
        TurnToHumanActionClient,
    )
    from sound_processing.enhance_audio import AudioEnhancement
except:
    has_ros = False

porcupine = None
pa = None
audio_stream = None
play_name = ""


FILT_LOW = 400
FILT_HIGH = 4000
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")

SAMPLE_RATE_WORK = 16000


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


class PorcupineDemo(Thread):
    """
    Demo class for wake word detection (aka Porcupine) library. It creates an input audio stream from a microphone,
    monitors it, and upon detecting the specified wake word(s) prints the detection time and index of wake word on
    console. It optionally saves the recorded audio into a file for further review.
    """

    _AUDIO_DEVICE_INFO_KEYS = ["index", "name", "defaultSampleRate", "maxInputChannels"]

    def __init__(
        self,
        # library_path,
        # model_file_path,
        keyword_file_paths,
        sensitivities,
        access_key,
        # input_device_index=None,
        output_path="None",
    ):

        """
        Constructor.

        :param library_path: Absolute path to Porcupine's dynamic library.
        :param model_file_path: Absolute path to the model parameter file.
        :param keyword_file_paths: List of absolute paths to keyword files.
        :param sensitivities: Sensitivity parameter for each wake word. For more information refer to
        'include/pv_porcupine.h'. It uses the
        same sensitivity value for all keywords.
        :param input_device_index: Optional argument. If provided, audio is recorded from this input device. Otherwise,
        the default audio input device is used.
        :param output_path: If provided recorded audio will be stored in this location at the end of the run.
        """

        super(PorcupineDemo, self).__init__()
        # self._library_path           = library_path
        # self._model_file_path        = model_file_path
        self._keyword_file_paths = keyword_file_paths
        self._sensitivities = sensitivities or 0.5
        # self._input_device_index     = input_device_index
        self.play_name = ""
        self.play_id = 0
        self.recorded_frames = Queue()
        self.__activate_vad_received = False
        self.__vad_enabled = True
        self.run_once = False
        self._access_key = access_key
        self._prevent_recording = False

        self._output_path = output_path
        if self._output_path is not None:
            self._recorded_frames_left = []
            self._recorded_frames_right = []

        if has_ros:
            print("Opening ros")
            rospy.init_node("vad", anonymous=True)
            print("Connecting to publisher")
            self.pub = rospy.Publisher("wav_send", String, queue_size=10)

            self.turn_to_human_client = TurnToHumanActionClient()

            self.sub_activate_vad = rospy.Subscriber(
                "/activate_vad", Bool, self.__activate_vad_callback
            )

            self.sub_vad_enabled = rospy.Subscriber(
                "vad_enabled", Bool, self.__vad_enabled_callback
            )
            self.run_once_sub = rospy.Subscriber(
                "/vad_run_once", Bool, self.__run_once_callback
            )
        else:
            self.pub = None
        print("Initialization done.")

    def __activate_vad_callback(self, data):
        print("activate_vad_received")
        self.__activate_vad_received = True

    def __vad_enabled_callback(self, data):
        if data.data == True:
            self.__vad_enabled = True
        elif data.data == False:
            self.__vad_enabled = False
        else:
            raise

    def __run_once_callback(self, data):
        self.run_once = True

    def process_channel(self, block):
        # pack(format, v1, v2, ...) - returns bytes object containing v1,v2,... packed according to format
        in_data = pack("<" + ("h" * len(block)), *block)

        b, a = butter_bandpass(FILT_LOW, FILT_HIGH, SAMPLE_RATE_WORK, order=5)
        zi_global = lfilter_zi(b, a)

        filtered_block, zi = lfilter(b, a, block, zi=zi_global)
        filtered_block = filtered_block.astype(np.int16)
        chunk_to_analyze = pack("<" + ("h" * len(filtered_block)), *filtered_block)
        return (in_data, chunk_to_analyze)

    def get_filtered_audio(self, in_data):

        decoded_block = np.fromstring(in_data, "Int16")
        # co drugi od poczatku
        channel_left = decoded_block[0::2]
        # co drugi od drugiego elementu (od elem z indexem)
        channel_right = decoded_block[1::2]

        (orig_left, filter_left) = self.process_channel(channel_left)
        (orig_right, filter_right) = self.process_channel(channel_right)

        return orig_left, filter_left, orig_right, filter_right

    def get_next_frame(self):
        if self.play_name == "":
            output = np.zeros(512 * 2 + 2, dtype=np.int16).tostring()
            self.play_id = 0
            return output

        output = self.sounds[self.play_name][
            self.play_id * 512 * 2 : (self.play_id + 1) * 512 * 2
        ]
        self.play_id = self.play_id + 1
        if len(output) < 512 * 2:
            output = np.pad(
                output, (0, (512 * 2) - len(output)), "constant", constant_values=(0, 0)
            )
            self.play_name = ""

        output = output.tostring()
        return output

    def audio_stream_callback(self, in_data, frame_count, time_info, status):
        if self._prevent_recording:
            return None, pyaudio.paContinue
        orig_left, filter_left, orig_right, filter_right = self.get_filtered_audio(
            in_data
        )
        self.recorded_frames.put(
            {
                "orig_l": orig_left,
                "filt_l": filter_left,
                "orig_r": orig_right,
                "filt_r": filter_right,
            }
        )

        output = self.get_next_frame()
        return output, pyaudio.paContinue

    def run(self):
        """
        Creates an input audio stream, initializes wake word detection (Porcupine) object, and monitors the audio
        stream for occurrences of the wake word(s). It prints the time of detection for each occurrence and index of
        wake word.
        """

        num_keywords = len(self._keyword_file_paths)

        keyword_names = list()
        for x in self._keyword_file_paths:
            keyword_names.append(
                os.path.basename(x)
                .replace(".ppn", "")
                .replace("_compressed", "")
                .split("_")[0]
            )

        print("listening for:")
        for keyword_name, sensitivity in zip(keyword_names, self._sensitivities):
            print("- %s (sensitivity: %f)" % (keyword_name, sensitivity))

        porcupine_l = None
        porcupine_l2 = None
        porcupine_r = None
        porcupine_r2 = None
        pa = None
        audio_stream = None

        # open the ON and OFF sound files for reading.
        wf = wave.open(os.path.join(DATA_DIR, "snd_on.wav"), "rb")
        wg = wave.open(os.path.join(DATA_DIR, "snd_off.wav"), "rb")
        try:
            # initialize porcupine module for each channel
            porcupine_r = pvporcupine.create(
                access_key=self._access_key,
                keyword_paths=self._keyword_file_paths,
            )
            porcupine_l = pvporcupine.create(
                access_key=self._access_key,
                keyword_paths=self._keyword_file_paths,
            )

            porcupine_l = pvporcupine.create(
                access_key=self._access_key, keyword_paths=self._keyword_file_paths
            )
            porcupine_r = pvporcupine.create(
                access_key=self._access_key, keyword_paths=self._keyword_file_paths
            )
            porcupine_l2 = pvporcupine.create(
                access_key=self._access_key, keyword_paths=self._keyword_file_paths
            )
            porcupine_r2 = pvporcupine.create(
                access_key=self._access_key, keyword_paths=self._keyword_file_paths
            )

            porcupine = porcupine_l

            print("sample rate", porcupine.sample_rate)
            print("frame len", porcupine.frame_length)

            pa = pyaudio.PyAudio()

            def open_audio_stream():
                audio_stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=2,
                    rate=porcupine.sample_rate,
                    input=True,
                    frames_per_buffer=porcupine.frame_length,
                )

            open_audio_stream()

            # open stream based on the wave object which has been input.
            wav_data = wf.readframes(-1)
            wav2_data = wg.readframes(-1)
            self.sounds = {
                "on": np.fromstring(wav_data, "Int16"),
                "off": np.fromstring(wav2_data, "Int16"),
            }

            last_detection_time = datetime.datetime.now()

            while True:
                try:
                    frame = self.recorded_frames.get(block=False)
                except:
                    continue

                #  callback
                pcm_l = frame["orig_l"]
                pcm_l = struct.unpack_from("h" * porcupine.frame_length, pcm_l)
                result_l = porcupine_l.process(pcm_l)

                pcm_r = frame["orig_r"]
                pcm_r = struct.unpack_from("h" * porcupine.frame_length, pcm_r)
                result_r = porcupine_r.process(pcm_r)

                pcm_l2 = frame["filt_l"]
                pcm_l2 = struct.unpack_from("h" * porcupine_l2.frame_length, pcm_l2)
                result_l2 = porcupine_l2.process(pcm_l2)

                pcm_r2 = frame["filt_r"]
                pcm_r2 = struct.unpack_from("h" * porcupine_r2.frame_length, pcm_r2)
                result_r2 = porcupine_r2.process(pcm_r2)

                # keyword_index = max(result_l, result_r)
                keyword_index = max(result_l, result_l2, result_r, result_r2)

                # will return True or -1
                if keyword_index >= 0:
                    detection_time = datetime.datetime.now()
                    time_diff = detection_time - last_detection_time

                    if time_diff.seconds >= 1:
                        last_detection_time = detection_time

                        print("Keyword detected.")

                        # Orient robot towards the human.
                        # if has_ros:
                        #     self._prevent_recording = True
                        #     self.turn_to_human_client.send_goal(TurnToHumanGoal())
                        #     self.turn_to_human_client.wait_for_result()
                        #     self._prevent_recording = False
                        # self.recorded_frames = Queue()

                        # record human voice
                        self.play_name = "on"
                        audio_stream.close()
                        self.runvad()
                        open_audio_stream()
                        self.play_name = "off"
                        self.__activate_vad_received = False

        finally:
            print("\nfinally")

            if porcupine_l is not None:
                porcupine_l.delete()

            if porcupine_r is not None:
                porcupine_r.delete()

            if porcupine_l2 is not None:
                porcupine_l2.delete()

            if porcupine_r2 is not None:
                porcupine_r2.delete()

            if audio_stream is not None:
                audio_stream.close()

            if pa is not None:
                pa.terminate()

    def runvad(self):
        THR_VOICED = 5  # start recording after that many voiced frames
        THR_UNVOICED = 8  # stop recording after that many unvoiced/silence frames
        THR_TIME = 5  # stop recording after that many seconds
        THR_POWER = 120  # minimum volume (power) for voiced frames
        DO_NORMALIZE = True  # do normalize the output wave

        RATE = SAMPLE_RATE_WORK
        CHUNK_DURATION_MS = 30  # supports 10, 20 and 30 (ms)
        PADDING_DURATION_MS = 1500  # 1 sec jugement
        NUM_PADDING_CHUNKS = int(PADDING_DURATION_MS / CHUNK_DURATION_MS)
        NUM_WINDOW_CHUNKS = int(400 / CHUNK_DURATION_MS)  # 400 ms / 30ms
        NUM_WINDOW_CHUNKS_END = NUM_WINDOW_CHUNKS * 5
        got_a_sentence = False

        # initialize ring buffer
        triggered = False

        def record_to_file(path, data, sample_width, rate):
            "Records from the microphone and outputs the resulting data to 'path'"
            print("-- Enter ")
            (channel_left, channel_right) = data
            wf = wave.open(path, "wb")
            wf.setnchannels(2)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            print("Pre loop")
            for left, right in zip(channel_left, channel_right):
                left_frame = pack("<h", left)
                wf.writeframes(left_frame)
                right_frame = pack("<h", right)
                wf.writeframes(right_frame)
            print("Post loop")
            wf.close()

        def normalize(snd_data):
            "Average the volume out"
            MAXIMUM = 32767  # 16384
            times = float(MAXIMUM) / max(abs(i) for i in snd_data)
            r = array("h")
            for i in snd_data:
                r.append(int(i * times))
            return r

        ring_buffer_left = collections.deque(maxlen=NUM_PADDING_CHUNKS)
        ring_buffer_flags_left = [0] * NUM_WINDOW_CHUNKS
        ring_buffer_index_left = 0
        ring_buffer_flags_end_left = [1] * NUM_WINDOW_CHUNKS_END
        ring_buffer_index_end_left = 0

        ring_buffer_right = collections.deque(maxlen=NUM_PADDING_CHUNKS)
        ring_buffer_flags_right = [0] * NUM_WINDOW_CHUNKS
        ring_buffer_index_right = 0
        ring_buffer_flags_end_right = [1] * NUM_WINDOW_CHUNKS_END
        ring_buffer_index_end_right = 0

        # initialize recording
        raw_data_left = array("h")
        raw_data_right = array("h")
        index = 0
        start_point = 0
        StartTime = time.time()
        TimeUse = 0
        cancelled = False
        print("* recording: ")

        num_unv = 0
        ignore = 5

        print("RUNVAD")

        while not got_a_sentence and TimeUse <= THR_TIME:
            if not self.__vad_enabled:
                cancelled = True
                break

            try:
                data = self.recorded_frames.get(block=False)
            except Exception as e:
                continue

            if ignore > 0:
                ignore = ignore - 1
                continue

            chunk_left = data["orig_l"]
            chunk_right = data["orig_r"]
            filtered_left = data["filt_l"][0:960]
            filtered_right = data["filt_r"][0:960]

            decoded_block_left = np.fromstring(filtered_left, "Int16")
            decoded_block_right = np.fromstring(filtered_right, "Int16")

            # add WangS
            raw_data_left.extend(array("h", chunk_left))
            raw_data_right.extend(array("h", chunk_right))
            index += len(decoded_block_left)
            TimeUse = time.time() - StartTime

            # check chunk for voice activity
            def check_activity(block):
                power = np.mean(np.abs(block))
                return power > THR_POWER

            active = check_activity(decoded_block_left) and check_activity(
                decoded_block_right
            )
            if active:
                num_unv = 0
            else:
                num_unv = num_unv + 1

            # display activity status
            print("O" if active else "-", end="")

            # update ring buffer's start and end flags
            def update_ring_buffer(
                ring_buffer,
                ring_buffer_flags,
                ring_buffer_flags_end,
                ring_buffer_index,
                ring_buffer_index_end,
                active,
                chunk,
            ):
                ring_buffer_flags[ring_buffer_index] = 1 if active else 0
                ring_buffer_index += 1
                ring_buffer_index %= NUM_WINDOW_CHUNKS
                ring_buffer_flags_end[ring_buffer_index_end] = 1 if active else 0
                ring_buffer_index_end += 1
                ring_buffer_index_end %= NUM_WINDOW_CHUNKS_END
                ring_buffer.append(chunk)

            update_ring_buffer(
                ring_buffer_left,
                ring_buffer_flags_left,
                ring_buffer_flags_end_left,
                ring_buffer_index_left,
                ring_buffer_index_end_left,
                active,
                chunk_left,
            )
            update_ring_buffer(
                ring_buffer_right,
                ring_buffer_flags_right,
                ring_buffer_flags_end_right,
                ring_buffer_index_right,
                ring_buffer_index_end_right,
                active,
                chunk_right,
            )

            # start point detection
            if not triggered:
                num_voiced_left = sum(ring_buffer_flags_left)
                num_voiced_right = sum(ring_buffer_flags_right)
                if num_voiced_left > THR_VOICED or num_voiced_right > THR_VOICED:
                    sys.stdout.write("<<")
                    ring_buffer_flags_end_left = [1] * NUM_WINDOW_CHUNKS_END
                    ring_buffer_flags_end_right = [1] * NUM_WINDOW_CHUNKS_END
                    triggered = True
                    start_point = index - 512 * 8  # start point
                    ring_buffer_left.clear()
                    ring_buffer_right.clear()
            else:
                if num_unv > THR_UNVOICED or TimeUse > THR_TIME:
                    sys.stdout.write(">>")
                    triggered = False
                    got_a_sentence = True

            sys.stdout.flush()
        sys.stdout.write("\n")

        if cancelled:
            print("* vad cancelled")
        else:
            print("* done recording")
            if True:
                got_a_sentence = False

                # write to file
                raw_data_left.reverse()
                raw_data_right.reverse()
                for index in range(start_point):
                    raw_data_left.pop()
                    raw_data_right.pop()
                raw_data_left.reverse()
                raw_data_right.reverse()

                if DO_NORMALIZE:
                    raw_data_left = normalize(raw_data_left)
                    raw_data_right = normalize(raw_data_right)

                now = datetime.datetime.now()
                fname = now.strftime("/tmp/%m-%d-%Y-%H-%M-%S") + ".wav"
                print(fname)
                record_to_file(fname, (raw_data_left, raw_data_right), 2, RATE)

                print("Saved to " + fname)
                if has_ros:
                    # Ensure the quality of the recorded audio is up to par.
                    AudioEnhancement(fname).enhance()
                    self.pub.publish(fname)
            if got_a_sentence:
                got_a_sentence = False
            else:
                pass


def main():
    access_key = "aDd541fQUB9+vb6KqcWV7kMBEvOkHQGV/bg7Z/1pbE1gcS0TmHzpYA=="
    keyword_file_paths = [
        "/home/wstyczen/tiago_public_ws/src/ros/dialogflow/Hey-Rico_en_linux_v2_1_0.ppn"
    ]
    # keyword_file_paths = os.path.join(RosPack().get_path('dialogflow'), 'Hey-Rico_en_linux_v2_1_0.ppn')
    sensitivities = [0.5]

    PorcupineDemo(
        # library_path=LIBRARY_PATH,
        # model_file_path=MODEL_FILE_PATH,
        keyword_file_paths=keyword_file_paths,
        access_key=access_key,
        sensitivities=sensitivities,
        # output_path='/tmp/out.wav',
        # input_device_index=DEVICE_ID
    ).run()


if __name__ == "__main__":
    main()
