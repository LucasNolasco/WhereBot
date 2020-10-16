#!/usr/bin/python3

import rospy
import requests
from std_msgs.msg import String
import os, socketio

base_url = "https://wherebot-backend.herokuapp.com"
#base_url = "http://localhost:8080"
robotID = "5f81dbc016aaa60021a0a48a"

def surveyCallback(data):
	image_paths = data.data.split(':')
	files = []
	
	for path in image_paths:
		imageName = path.split('/')[-1]
		imageType = path.split('.')[-1]
		files.append( ('files', (imageName, open(path, 'rb'), f'image/{imageType}' )) )

	reqBody1 = {
		"info": "Info",
		"robotID": robotID,
	}

	reqBody2 = {
		"state": "Stopped",
		"robotID": robotID
	}


	try:
		response = requests.post(base_url + "/survey/new", files=files, data=reqBody1)
		requests.post(base_url + "/robot/setstate", data=reqBody2)
		print("POST Request to create new survey done: ", response.json())
	except ValueError:
		print("POST Request to create new survey error: ", ValueError)
			

def stateCallback(data):
	if data.data not in ["Doing", "Finished"]:
		return
	
	reqBody = {
		"state": data.data,
		"robotID": robotID
	}

	try:
		requests.post(base_url + "/robot/setstate", data=reqBody)
		print("POST Request to change state done.")
	except ValueError:
		print("POST Request to change state error: ", ValueError)
	

rospy.init_node("server")
rate = rospy.Rate(0.2)
state = String()

pub = rospy.Publisher('/survey/state', String, queue_size=10)

rospy.Subscriber("/survey/finish/data", String, surveyCallback)
rospy.Subscriber("/survey/state", String, stateCallback)

# Socket connection
sio = socketio.Client()
sio.connect(base_url)

@sio.on(robotID)
def on_message(data):
	print("SOCKET data received: ", data)
	if data["state"] == "Start":
		state.data = "Start"
		pub.publish(state)

rospy.spin()
