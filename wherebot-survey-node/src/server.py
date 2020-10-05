#!/usr/bin/python3

import rospy
import requests
from std_msgs.msg import String
import os

base_url = "https://wherebot-backend.herokuapp.com"
robotID = "5f7a3474590c2d0021953866"

def surveyCallback(data):
	image_paths = data.data.split('-')
	files = []
	
	for path in image_paths:
		imageName = path.split('/')[-1]
		imageType = path.split('.')[-1]
		files.append( ('files', (imageName, open(path, 'rb'), f'image/{imageType}' )) )

	reqBody = {
		"info": "Info",
		"robotID": robotID,
	}

	try:
		response = requests.post(base_url + "/survey/new", files=files, data=reqBody)
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


while not rospy.is_shutdown():
	try:
		r = requests.get(url = base_url+"/robot/getstate", json={ "robotID": robotID }) 
		dados = r.json()
		print("GET Request state done: ", dados)
		
		if dados["state"] == "Start":
			state.data = "Start"
			pub.publish(state)

	except ValueError:
		print("GET Request state error: ", ValueError)
	
	rate.sleep()


#/home/wagner/Pictures/heatmap1.jpeg-/home/wagner/Pictures/heatmap2.jpeg-/home/wagner/Pictures/heatmap3.jpeg