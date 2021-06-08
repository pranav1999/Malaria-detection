import tkinter  
import cv2,os
import numpy as np
import csv
import glob

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
import joblib


m = tkinter.Tk() 
''' 
widgets are added here 

'''


# generating parasitized 
def generate():
	label = "parasitized"
	dirlist = glob.glob("cell_images/"+label+"/*.png")
	file = open("csv/dataset.csv","a")

	for img_path in dirlist:
		im = cv2.imread(img_path)

		im = cv2.GaussianBlur(im,(5,5),2)


		im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

		ret,thresh = cv2.threshold(im_gray,127,255,0)
		contours,_ = cv2.findContours(thresh,1,2)

		file.write(label)
		file.write(",")

		for i in range(5):
			try:
				area = cv2.contourArea(contours[i])
				file.write(str(area))
			except:
				file.write("0")

			file.write(",")

		file.write("\n")

	cv2.waitKey(19000)


# generating uninfected


def generate_u():
	label = "uninfected"
	dirlist = glob.glob("cell_images/"+label+"/*.png")
	file = open("csv/dataset.csv","a")

	for img_path in dirlist:
		im = cv2.imread(img_path)

		im = cv2.GaussianBlur(im,(5,5),2)


		im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

		ret,thresh = cv2.threshold(im_gray,127,255,0)
		contours,_ = cv2.findContours(thresh,1,2)

		file.write(label)
		file.write(",")

		for i in range(5):
			try:
				area = cv2.contourArea(contours[i])
				file.write(str(area))
			except:
				file.write("0")

			file.write(",")

		file.write("\n")

	cv2.waitKey(19000)


# genrate result

def result():
	dataframe = pd.read_csv("csv/dataset.csv")
	#print(dataframe.head())

	#step2:split into training and test data

	x = dataframe.drop(["Label"],axis=1)
	y = dataframe["Label"]
	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)


	#step4: build a model

	model = RandomForestClassifier(n_estimators=100, max_depth=5)
	model.fit(x_train,y_train)
	joblib.dump(model,"rf_malaria_100_5")

	## step5: make predictions and get classification report

	predictions = model.predict(x_test)

	print(metrics.classification_report(predictions,y_test))
	print(model.score(x_test,y_test))

	






m.geometry("400x400")
m.title("Malaria Detection")




but= tkinter.Button(text="generate", command=generate)
but.pack()

but1= tkinter.Button(text="generate_u", command=generate_u)
but1.pack()

but2=tkinter.Button(text="report", command=result)
but2.pack()

m.mainloop() 
