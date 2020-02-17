#!/usr/bin/env python
# encoding: utf8

import threading
import rospy
import actionlib
from std_msgs.msg import String, Bool
import tiago_msgs.msg

import time
import tempfile

#from sound_play.msg import SoundRequest
#from sound_play.libsoundplay import SoundClient

from Levenshtein import distance

import pygame.mixer

import copy
import sys
import os
import codecs

import pl_nouns.odmiana as ro



import pyaudio
import wave

def play_sound(fname, start_pos):
    try:
        # open wave file
        wave_file = wave.open(fname, 'rb')

        # initialize audio
        py_audio = pyaudio.PyAudio()
        stream = py_audio.open(format=py_audio.get_format_from_width(wave_file.getsampwidth()),
                               channels=wave_file.getnchannels(),
                               rate=wave_file.getframerate(),
                               output=True)

        # skip unwanted frames
        n_frame_start = int(start_pos * wave_file.getframerate())
        wave_file.setpos(n_frame_start)

        # write desired frames to audio buffer
        n_frames = wave_file.getnframes() - n_frame_start + 1 #int(length * wave_file.getframerate())
        frames = wave_file.readframes(n_frames)
        stream.write(frames)

        # close and terminate everything properly
        stream.close()
        py_audio.terminate()
        wave_file.close()
    except Exception as e:
        print e

# Action server for speaking text sentences
class SaySentenceActionServer(object):
    def __init__(self, name, playback_queue, odm, sentence_dict, agent_name, cred_file):
        self._action_name = name
        self._playback_queue = playback_queue
        self._odm = odm
        self._sentence_dict = sentence_dict
        self._agent_name = agent_name
        self._cred_file = cred_file
        self._as = actionlib.SimpleActionServer(self._action_name, tiago_msgs.msg.SaySentenceAction, execute_cb=self.execute_cb, auto_start = False)
        self._as.start()

    def execute_cb(self, goal):
        # create messages that are used to publish feedback/result
        _feedback = tiago_msgs.msg.SaySentenceFeedback()
        _result = tiago_msgs.msg.SaySentenceResult()

        sentence_uni = goal.sentence.decode('utf-8')
        sentence_uni = self._odm.odmien(sentence_uni)

        prefix = u'niekorzystne warunki pogodowe'
        prefix_time_length = 2.1
        ss = strip_inter(sentence_uni).strip().upper()
        if sentence_uni.startswith( prefix ):
            pub_txt_msg.publish( sentence_uni[len(prefix):] )
            print u'detected "' + prefix + u'"'
            response, sound_fname = detect_intent_text(self._agent_name, "test_sess_012", ss.lower(), "pl", self._cred_file)
            sound_fname = (sound_fname[0], sound_fname[1], prefix_time_length)     # Cut out the prefix
            print 'received response:', response.query_result
            best_d = 0
            best_k = None
        else:
            pub_txt_msg.publish( sentence_uni )
            best_k = ""
            best_d = 999
            print "Searching best match for", ss 
            for k in self._sentence_dict.keys():
                #print k, v
                d = distance(k, ss)
            #    print k, d
                if d < best_d:
                    best_d = d
                    best_k = k
            sound_fname = (self._sentence_dict[best_k], 'keep', 0.0)

        print u'Starting action for "' + sentence_uni + u'"'
        success = True
        if best_d < 5:
            print "Wiem co powiedzieć!", ss, best_k, sound_fname
            sound_id = self._playback_queue.addSound( sound_fname )

            while not self._playback_queue.finishedSoundId(sound_id) and not rospy.is_shutdown():
                # check that preempt has not been requested by the client
                if self._as.is_preempt_requested():
                    rospy.loginfo('%s: Preempted' % self._action_name)
                    self._as.set_preempted()
                    success = False
                    break
                self._as.publish_feedback(_feedback)
                rospy.sleep(0.1)

        if success:
            rospy.loginfo('%s: Succeeded' % self._action_name)
            _result.success = True
            self._as.set_succeeded(_result)

        print u'Ended action for "' + sentence_uni + u'"', success

def detect_intent_audio(project_id, session_id, audio_file_path, language_code, cred_file):
    """Returns the result of detect intent with an audio file as input.

    Using the same `session_id` between requests allows continuation
    of the conversation."""
    import dialogflow_v2 as dialogflow
    print 'detect_intent_audio'
    session_client = dialogflow.SessionsClient.from_service_account_file(cred_file)

    # Note: hard coding audio_encoding and sample_rate_hertz for simplicity.
    audio_encoding = dialogflow.enums.AudioEncoding.AUDIO_ENCODING_LINEAR_16
    sample_rate_hertz = 16000

    session = session_client.session_path(project_id, session_id)
    print('Session path: {}\n'.format(session))

    with open(audio_file_path, 'rb') as audio_file:
        input_audio = audio_file.read()

    audio_config = dialogflow.types.InputAudioConfig(
        audio_encoding=audio_encoding, language_code=language_code,
        sample_rate_hertz=sample_rate_hertz)
    # Set the query parameters with sentiment analysis
    voice = dialogflow.types.VoiceSelectionParams(
        name = "pl-PL-Wavenet-B"
    )
    synt = dialogflow.types.SynthesizeSpeechConfig(
        pitch = -10,
        speaking_rate = 0.8,
        voice=voice
    )
    output_audio_config = dialogflow.types.OutputAudioConfig(
        audio_encoding=audio_encoding,
        synthesize_speech_config=synt
    )
    
    query_input = dialogflow.types.QueryInput(audio_config=audio_config)

    response = session_client.detect_intent(
        session=session, query_input=query_input,
        input_audio=input_audio,
        output_audio_config=output_audio_config
    )

#    print('=' * 20)
#    print(u'Query text: {}'.format(response.query_result.query_text))
#    print(u'Detected intent: {} (confidence: {} params: {})'.format(
#        response.query_result.intent.display_name,
#        response.query_result.intent_detection_confidence,
#        response.query_result.parameters))
#    print(u'Fulfillment text: {}\n'.format(
#        response.query_result.fulfillment_text))

    #with open('/tmp/output.wav', 'wb') as out:
    #    out.write(response.output_audio)
    #    print('Audio content written to file "output.wav"')

    print 'detect_intent_audio resp', response

    out = tempfile.NamedTemporaryFile(delete=False)
    out.write(response.output_audio)
    fname = out.name
    out.close()

    return response, (fname, 'delete', 0.0)

def detect_intent_text(project_id, session_id, text, language_code, cred_file):
    """Returns the result of detect intent with an audio file as input.

    Using the same `session_id` between requests allows continuation
    of the conversation."""
    import dialogflow_v2 as dialogflow

    session_client = dialogflow.SessionsClient.from_service_account_file(cred_file)

    # Note: hard coding audio_encoding and sample_rate_hertz for simplicity.
    audio_encoding = dialogflow.enums.AudioEncoding.AUDIO_ENCODING_LINEAR_16
    sample_rate_hertz = 16000

    session = session_client.session_path(project_id, session_id)
    print('Session path: {}\n'.format(session))

    audio_config = dialogflow.types.InputAudioConfig(
        audio_encoding=audio_encoding, language_code=language_code,
        sample_rate_hertz=sample_rate_hertz)
    # Set the query parameters with sentiment analysis
    voice = dialogflow.types.VoiceSelectionParams(
        name = "pl-PL-Wavenet-B"
    )
    synt = dialogflow.types.SynthesizeSpeechConfig(
        pitch = -10,
        speaking_rate = 0.8,
        voice=voice
    )
    output_audio_config = dialogflow.types.OutputAudioConfig(
        audio_encoding=audio_encoding,
        synthesize_speech_config=synt
    )
    
    text_input = dialogflow.types.TextInput(text=text, language_code=language_code)
    query_input = dialogflow.types.QueryInput(text=text_input)

    response = session_client.detect_intent(
        session=session, query_input=query_input,
        output_audio_config=output_audio_config
    )

#    print('=' * 20)
#    print(u'Query text: {}'.format(response.query_result.query_text))
#    print(u'Detected intent: {} (confidence: {} params: {})'.format(
#        response.query_result.intent.display_name,
#        response.query_result.intent_detection_confidence,
#        response.query_result.parameters))
#    print(u'Fulfillment text: {}\n'.format(
#        response.query_result.fulfillment_text))

    #with open('/tmp/output.wav', 'wb') as out:
    #    out.write(response.output_audio)
    #    print('Audio content written to file "output.wav"')
    out = tempfile.NamedTemporaryFile(delete=False)
    out.write(response.output_audio)
    fname = out.name
    out.close()
    print('Audio content written to temporary file')
        
    return response, (fname, 'delete', 0.0)

#detect_intent_audio("fiery-set-259318", "test_sess_01", sys.argv[1], "pl")


pub_txt_msg = rospy.Publisher('txt_msg', String, queue_size=10)
pub_txt_voice_cmd_msg = rospy.Publisher('txt_voice_cmd_msg', String, queue_size=10)
pub_cmd = rospy.Publisher('rico_cmd', tiago_msgs.msg.Command, queue_size=10)
pub_vad_active = rospy.Publisher('vad_active', Bool, queue_size=10)

#soundhandle = SoundClient()

class PlaybackQueue:
    def __init__(self):
        self.__queue__ = []
        self.__queue_lock__ = threading.Lock()
        self.__sound_id__ = 0
        self.__current_sound_id__ = None

    def spin_once(self):
        fname = None
        self.__queue_lock__.acquire()
        if bool(self.__queue__):
            sound_id, (fname, keep_mode, start_time) = self.__queue__.pop(0)
            self.__current_sound_id__ = sound_id
        self.__queue_lock__.release()
        if fname:
            self.__playBlockingsound__(fname, start_time)
            self.__queue_lock__.acquire()
            self.__current_sound_id__ = None
            self.__queue_lock__.release()
            if keep_mode == 'delete':
                os.remove(fname)
            elif keep_mode == 'keep':
                # do nothing
                pass
            else:
                raise Exception('Wrong keep_mode: "' + keep_mode + '"')

    def finishedSoundId(self, sound_id):
        result = True
        self.__queue_lock__.acquire()
        if self.__current_sound_id__ == sound_id:
            result = False
        else:
            for s_id, _ in self.__queue__:
                if s_id == sound_id:
                    result = False
                    break
        self.__queue_lock__.release()
        return result

    def addSound(self, sound_file):
        assert sound_file[1] == 'keep' or sound_file[1] == 'delete'
        self.__queue_lock__.acquire()
        self.__queue__.append( (self.__sound_id__, sound_file) )
        result_sound_id = self.__sound_id__
        self.__sound_id__ = self.__sound_id__ + 1
        self.__queue_lock__.release()
        return result_sound_id

    def __playBlockingsound__(self, fname, start_time):
        pub_vad_active.publish(False)
        print 'playBlockingsound: BEGIN'
        print '  file:', fname
        #soundhandle.playWave(fname, 1, blocking=True)

        '''
        SONG_END = pygame.USEREVENT + 1
        pygame.mixer.music.set_endevent(SONG_END)
        pygame.mixer.music.load(fname)
        pygame.mixer.music.play(0)

        timeout = time.time() + 15   # 15 seconds from now

        finished = False
        while not finished:
            for event in pygame.event.get():
                if event.type == SONG_END:
                    print "the song ended!"
                    finished = True
            if time.time() > timeout:
                print "timeout - assuming the sound play ended"
                finished = True
            time.sleep(0.1)
        '''

        play_sound(fname, start_time)
        print 'playBlockingsound: END'

        pub_vad_active.publish(True)

def callback_common(response, sound_file, playback_queue):
    print response.query_result

    cmd = tiago_msgs.msg.Command()
    cmd.query_text = response.query_result.query_text
    cmd.intent_name = response.query_result.intent.name
    for param_name, param in response.query_result.parameters.fields.iteritems():

        param_str = unicode(param)
        colon_idx = param_str.find(':')
        param_type = param_str[0:colon_idx]
        assert param_type == 'string_value'
        param_value = param_str[colon_idx+1:].strip()[1:-1]

        print 'param_name: "' + param_name + '"'
        print 'param_type: "' + param_type + '"'
        print 'param_value: "' + param_value + '"'

        cmd.param_names.append( param_name )
        cmd.param_values.append( param_value )

    cmd.confidence = response.query_result.intent_detection_confidence
    cmd.response_text = response.query_result.fulfillment_text
    pub_cmd.publish(cmd)

    if len(response.query_result.fulfillment_text) > 0:
        pub_txt_msg.publish(response.query_result.fulfillment_text)
        playback_queue.addSound(sound_file)

def callback(data, agent_name, playback_queue, cred_file):
    rospy.loginfo("I heard %s", data.data)
    response, sound_file = detect_intent_text(agent_name, "test_sess_012", data.data, "pl", cred_file)
    callback_common(response, sound_file, playback_queue)

def callback_wav(data, agent_name, playback_queue, cred_file):
    rospy.loginfo("I recorded %s", data.data)
    response, sound_file = detect_intent_audio(agent_name, "test_sess_012", data.data, "pl", cred_file)
    pub_txt_voice_cmd_msg.publish(response.query_result.query_text)
    callback_common(response, sound_file, playback_queue)

class Odmieniacz:
    def __init__(self):
        self.o = ro.OdmianaRzeczownikow()

    def przypadki(self, word, przyp):
        blocks = self.o.getBlocks(word)
        if len(blocks) == 0:
            word_m = word
            lp = True
        else:
            m_lp = self.o.getMianownikLp(blocks)
            if len(m_lp) == 0:
                m_lm = self.o.getMianownikLm(blocks)
                word_m = m_lm[0]
                lp = False
            else:
                word_m = m_lp[0]
                lp = True

        if przyp == 'mianownik':
            word_p = word

        if przyp == 'biernik':
            if lp:
                word_p = self.o.getBiernikLp(blocks, mianownik=word_m)
                if len(word_p) == 0:
                    word_p = word_m
                else:
                    word_p = word_p[0]
            else:
                word_p = self.o.getBiernikLm(blocks, mianownik=word_m)
                if len(word_p) == 0:
                    word_p = word_m
                else:
                    word_p = word_p[0]

        if przyp == 'dopelniacz':
            if lp:
                word_p = self.o.getDopelniaczLp(blocks, mianownik=word_m)
                if len(word_p) == 0:
                    word_p = word_m
                else:
                    word_p = word_p[0]
            else:
                word_p = self.o.getDopelniaczLm(blocks, mianownik=word_m)
                if len(word_p) == 0:
                    word_p = word_m
                else:
                    word_p = word_p[0]

        return word_m, word_p

    def odmien(self, s):
        result = copy.copy(s)
        while True:
            l_brace_idx = result.find('{')
            if l_brace_idx < 0:
                break
            r_brace_idx = result.find('}', l_brace_idx)
            if r_brace_idx < 0:
                break
            odm = result[l_brace_idx+1:r_brace_idx]
            print odm
            quot_idx1 = odm.find('"')
            quot_idx2 = odm.find('"', quot_idx1+1)
            word_orig = odm[quot_idx1+1:quot_idx2]
            print word_orig
            sep_idx = odm.find(',', quot_idx2+1)
            przyp = odm[sep_idx+1:].strip()
            word_m, word_p = self.przypadki(word_orig, przyp)
            result = result[0:l_brace_idx] + word_p + result[r_brace_idx+1:]
        return result

def strip_inter(string):
    return string.replace(".", "").replace(",","")

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('talker', anonymous=True)
    pygame.init()

    odm = Odmieniacz()
    playback_queue = PlaybackQueue()

    agent_name = rospy.get_param('~agent_name')
    data_dir = rospy.get_param('~data_dir')

    sentence_dict = {}
    from itertools import izip
    for sent, fname in izip(codecs.open(os.path.join(data_dir, "labels.txt"), encoding="utf-8"), open(os.path.join(data_dir, "files.txt"))):
        ss = unicode(sent) 
        sentence_dict[strip_inter(ss).strip().upper()] = os.path.join(data_dir, fname.strip())
    '''
    rospy.sleep(1)
    key0 = sentence_dict.keys()[0]
    fname = sentence_dict[key0]
    print 'fname', fname
    play_sound(fname, 0.5)
    print 'end'
    #pygame.mixer.music.load(fname)
    #pygame.mixer.music.play(0, 0.5)
    rospy.sleep(2)
    exit(0)
    '''
    #text = raw_input('.')
    #response, sound_fname = detect_intent_text(agent_name, "test_sess_012", u'niekorzystne warunki pogodowe ' + text, "pl")
    #sound_fname = (sound_fname[0], sound_fname[1], 0.6)
    #play_sound(sound_fname[0], 0.0)
    #print 'received response:', response.query_result
    #exit(0)

    if not 'GOOGLE_CREDENTIALS_INCARE_DIALOG' in os.environ:
        raise Exception('Env variable "GOOGLE_CREDENTIALS_INCARE_DIALOG" is not set')
    if not 'GOOGLE_CREDENTIALS_TEXT_TO_SPEECH' in os.environ:
        raise Exception('Env variable "GOOGLE_CREDENTIALS_TEXT_TO_SPEECH" is not set')

    cred_file_incare_dialog = os.environ['GOOGLE_CREDENTIALS_INCARE_DIALOG']
    cred_file_text_to_speech = os.environ['GOOGLE_CREDENTIALS_TEXT_TO_SPEECH']

    rospy.Subscriber("txt_send", String, lambda x: callback(x, agent_name, playback_queue, cred_file_incare_dialog))

    rospy.Subscriber("wav_send", String, lambda x: callback_wav(x, agent_name, playback_queue, cred_file_incare_dialog))

    say_as = SaySentenceActionServer( 'rico_says', playback_queue, odm, sentence_dict, 'text-to-speech-qtruau', cred_file_text_to_speech)

    # spin() simply keeps python from exiting until this node is stopped
    while not rospy.is_shutdown():
        playback_queue.spin_once()
        rospy.sleep(0.1)

if __name__ == '__main__':
    listener()
