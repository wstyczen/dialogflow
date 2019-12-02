#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from tiago_msgs.msg import Command

# encoding: utf8

import sys

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
    pub.publish(response.query_result.fulfillment_text);

    print response.query_result

    cmd = Command()
    cmd.query_text = response.query_result.query_text
    cmd.intent_name = response.query_result.intent.name
    for param_name, param in response.query_result.parameters.fields.iteritems():

        param_str = str(param)
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

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('talker', anonymous=True)

    agent_name = rospy.get_param('~agent_name')

    rospy.Subscriber("txt_send", String, lambda x: callback(x, agent_name))

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
