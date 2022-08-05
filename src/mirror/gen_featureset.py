#!/usr/bin/python
# -*- coding: utf-8 -*-

from pickle import NONE
from random import randint
import numpy as np
import pandas as pd
import mediapipe as mp
import cv2


class Gen_Featureset():

    def __init__(self) -> None:
        

        self.df = pd.read_csv('./src/mirror/dataset/dataset.csv')

        print("all data size",self.df.shape)



    def get_landmarks(self,image_path):

        IMG = cv2.imread(image_path)

        face_mesh = mp.solutions.mediapipe.python.solutions.face_mesh

        with face_mesh.FaceMesh(
        static_image_mode=True,max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as face:

            landmarks = face.process(cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)) # get the multi_face_landmarks named tuple 因为有可能有多个脸。所以先获得一个列表
            #print(len(self.result.multi_face_landmarks[0].landmark)) #在列表中的landmark字典里 储存着landmark

        return landmarks

        
    def draw_landmarks(self,image_path ,landmarks):

        IMG = cv2.imread(image_path)

        drawing_utils = mp.solutions.mediapipe.python.solutions.drawing_utils
        drawing_utils.draw_landmarks(IMG , landmarks.multi_face_landmarks[0] , mp.solutions.mediapipe.python.solutions.face_mesh.FACEMESH_TESSELATION) # drawing dots and edges

        cv2.namedWindow("img",0)
        cv2.resizeWindow("img", IMG.shape[1] // 4, IMG.shape[0] // 4) #adjust window's width and height
        cv2.imshow('img',IMG)
        cv2.waitKey(0)

    def poly_area(self,vtx_list,idx_list):   

        '''        
        计算面积

        Implementation of Shoelace formula
        see: https://stackoverflow.com/a/30408825

        '''

        pts = []

        for idx in idx_list:
            pts.append([vtx_list[idx].x , vtx_list[idx].y ])

        x = np.array(pts)[:,[0]].flatten()
        y = np.array(pts)[:,[1]].flatten()
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))



    def calculate_eyelid_area(self,image):

        '''特征工程 上下眼皮围起来的面积'''

        landmarks = self.get_landmarks(image)

        area = 0
        

        left_up_eye_point =     [398,384,385,386,387,388,466]
        left_bottom_eye_point = [382,381,380,374,373,390,249]
        right_up_eye_point =     [246,161,160,159,158,157,173]
        right_bottom_eye_point = [7,163,144,145,153,154,155]



        
        #计算上下眼皮围起来的面积
        
        lf_eye_index =   [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263]  
        rt_eye_index =   [133, 155, 154, 153, 145, 144, 163, 7, 33, 246, 161, 160, 159, 158, 157, 173, 133] 



        if(landmarks.multi_face_landmarks): # success get the face  
            

            lf_eye_area = self.poly_area(landmarks.multi_face_landmarks[0].landmark,lf_eye_index) # 左眼睛多边形的面积
            rt_eye_area = self.poly_area(landmarks.multi_face_landmarks[0].landmark,rt_eye_index) # 右眼睛多边形的面积

            area = lf_eye_area + rt_eye_area

            area = (area * 1000)*(area * 1000) # normalize the feature 


        else: # error on getting the face

            area = -1.0

        return area 

    
    def merge_blink_feature(self): 
        '''合并特征'''

        feature_arr = []

        blink_series = self.df.loc[:,"image_path"].str.contains("eye_closed") # str.contains() return a bool series
        neutral_series = self.df.loc[:,"image_path"].str.contains("neutral") # str.contains() return a bool series

        blink_df = self.df.loc[blink_series] #获得 闭眼表情 dataframe
        neutral_df = self.df.loc[neutral_series] #获得 自然表情 dataframe

        print("blink imageset size",blink_df.shape)
        print("neutral imageset size",neutral_df.shape)        

        for i,image_path in enumerate(blink_df.loc[:,"image_path"]): 

            if(i% 500 ==0):
                print(i)           

            area = self.calculate_eyelid_area(image_path) #calculate the feature :distance between eyelids

            if(area>=0): #by pass some error data  
                
                feature = {"image_path":image_path,"area":area,"eyeblink":1.0}                
                feature_arr.append(feature)


        for i,image_path in enumerate(neutral_df.loc[:,"image_path"]):

            if(i% 500 ==0):
                print(i)           

            area = self.calculate_eyelid_area(image_path)    

            if(area>=5.0): #by pass some error data        
            
                feature = {"image_path":image_path,"area":area,"eyeblink":0.0}
                feature_arr.append(feature)
        

        print(feature_arr)
        return feature_arr



    def write_csv(self,features_arr,csv_file):

        df = pd.DataFrame(features_arr)

        df.to_csv(csv_file)


if (__name__ == '__main__'):

    gen_featureset = Gen_Featureset()

    #show image with landmark

    def drawlandmark():

        idx = randint(0,gen_featureset.df.shape[0]) # pick a random face
        image_path = gen_featureset.df.loc[idx,'image_path'] # get the image path
        lm = gen_featureset.get_landmarks(image_path) # get the landmarks
        gen_featureset.draw_landmarks(image_path , lm) #draw the landmarks
    drawlandmark()


    # generate the featureset
    '''
    df = gen_featureset.merge_blink_feature()
    gen_featureset.write_csv( df , "./src/mirror/dataset/featureset.csv")


    import matplotlib.pyplot as plt

    df = pd.read_csv("./src/mirror/dataset/featureset.csv")

    print(df.head)

    plt.scatter(df.loc[:,"area"],df.loc[:,"eyeblink"])
    plt.show()
    '''

