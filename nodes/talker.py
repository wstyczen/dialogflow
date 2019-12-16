#!/usr/bin/env python
# encoding: utf8

import rospy
from std_msgs.msg import String
from tiago_msgs.msg import Command


import copy
import sys

import pl_nouns.odmiana as ro

def detect_intent_audio(project_id, session_id, audio_file_path, language_code):
    """Returns the result of detect intent with an audio file as input.

    Using the same `session_id` between requests allows continuation
    of the conversation."""
    import dialogflow_v2 as dialogflow

    session_client = dialogflow.SessionsClient()

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

    with open('output.wav', 'wb') as out:
        out.write(response.output_audio)
        print('Audio content written to file "output.wav"')

    return response


def detect_intent_text(project_id, session_id, text, language_code):
    """Returns the result of detect intent with an audio file as input.

    Using the same `session_id` between requests allows continuation
    of the conversation."""
    import dialogflow_v2 as dialogflow

    session_client = dialogflow.SessionsClient()

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

    with open('output.wav', 'wb') as out:
        out.write(response.output_audio)
        print('Audio content written to file "output.wav"')
        
    return response

#detect_intent_audio("fiery-set-259318", "test_sess_01", sys.argv[1], "pl")


pub = rospy.Publisher('txt_msg', String, queue_size=10)
pub_cmd = rospy.Publisher('tiago_cmd', Command, queue_size=10)

def callback(data, agent_name):
    rospy.loginfo("I heard %s", data.data)
    response = detect_intent_text(agent_name, "test_sess_012", data.data, "pl")
    if len(response.query_result.fulfillment_text) > 0:
        pub.publish(response.query_result.fulfillment_text);

    print response.query_result

    cmd = Command()
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

def callback_wav(data, agent_name):
    rospy.loginfo("I recorded %s", data.data)
    response = detect_intent_audio(agent_name, "test_sess_012", data.data, "pl")
    if len(response.query_result.fulfillment_text) > 0:
        pub.publish(response.query_result.fulfillment_text)

    print response.query_result

    cmd = Command()
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

odm = Odmieniacz()

def callbackRicoSays(data, agent_name):
    global odm
    data_uni = data.data.decode('utf-8')
    data_uni = odm.odmien(data_uni)
    pub.publish(data_uni)

def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('talker', anonymous=True)

    agent_name = rospy.get_param('~agent_name')

    rospy.Subscriber("txt_send", String, lambda x: callback(x, agent_name))

    rospy.Subscriber("wav_send", String, lambda x: callback_wav(x, agent_name))

    rospy.Subscriber("rico_says", String, lambda x: callbackRicoSays(x, agent_name))

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
