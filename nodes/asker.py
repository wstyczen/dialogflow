#!/usr/bin/env python
# encoding: utf8

import rospy
from tiago_msgs.msg import Command
import actionlib
from repeat_action_server.msg import AskToRepeatAction, AskToRepeatGoal

INTENT_NAME = "projects/incare-dialog-agent/agent/intents/2f028022-05b6-467d-bcbe-e861ab449c17"
client = actionlib.SimpleActionClient('/repeat_action', AskToRepeatAction)
        
def callback(cmd):
    if cmd.intent_name == INTENT_NAME:
        goal = AskToRepeatGoal()
        client.send_goal(goal)

rospy.init_node('asker', anonymous=True)
rospy.Subscriber('/rico_cmd', Command, lambda x: callback(x))
rospy.spin()
