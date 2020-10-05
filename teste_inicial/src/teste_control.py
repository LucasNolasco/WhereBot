#!/usr/bin/env python

import rospy
from std_msgs.msg import String

def state_callback(data):
    if data.data == "Doing":
        rospy.loginfo("Exploration started")

if __name__ == '__main__':
    rospy.init_node("teste_control", anonymous=False)

    server_publisher = rospy.Publisher("/survey/state", String, queue_size=10)
    rospy.Subscriber("/survey/state", String, state_callback)

    while not rospy.is_shutdown():
        userInput = input("Message to send: ")
        msg = String()
        msg.data = userInput

        server_publisher.publish(msg)
