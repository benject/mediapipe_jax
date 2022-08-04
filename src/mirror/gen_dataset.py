#!/usr/bin/python
# -*- coding: utf-8 -*-

import os,sys
import json

import pandas as pd

'''
read the facescape dataset from the front view camera list to get the corresponding useful image as dataset
save the dataset to csv
'''

class Gen_Dataset():

    def __init__(self,params_file) -> None: 
        
        with open(params_file,"r") as f:        
            self.params = json.load(f)       #read the paramters in config json file
        
    def get_dataset(self):

        image_folder = self.params['image_folder']

        ids_list = os.listdir(image_folder) # ["1".."359"]

        #第二套拍摄方案，相机的摆放角度有变化

        ids2 = ["11","14","15","20","25","28","32","39","40","41","42","49","50","54","62","68","75","78","84","88","92","97","157","167","169","173","174","178","283","298",
                "101","108","113","115","116","125","131","134","137","138","181","186","203","205","216","218","221","222","223","227","230","237","304","306","309","312","313","316","328","331","334","345","358","359"]

        ids1 = list(set(ids_list) - set(ids2))

        #print(len(ids1))
        #print(len(ids2))


        ids_folder = [ os.path.join(image_folder,id) for id in os.listdir(image_folder)] #get all ids folder [1...359] 
        #print(len(ids_folder)) # 359
        

        exps_folder = []

        for id_folder in ids_folder:
            
            exps = os.listdir(id_folder)

            for exp in exps:

                exp_folder = os.path.join(id_folder,exp)
                exps_folder.append(exp_folder)  #get all exps folder  #len: 359*20 = 7180

        print("starting handle the exp image")


        dataset = []
        
        n = 0        
        
        for exp_folder in exps_folder:
            
            n = n+1

            id = exp_folder.split("\\")[4] #from path get id 

            #print(id)

            params_json_file = os.path.join(exp_folder,'params.json')

            #print(params_json)

            with open(params_json_file,'r') as f: #each exp imageset has one param.json file , we need to get the 4 front view of the image set.

                exp_img_param = json.load(f)
            
            final_image_path = ''
       
            if id in ids1:
                
                for key in exp_img_param:

                    if ("sn" in key) and (exp_img_param[key] in self.params["cam_sn_list"]):

                        final_image_name = key.split("_")[0] + ".jpg" # get the correct actual image ## cam_sn_list -> xx_sn -> xx
                        final_image_path = os.path.join(exp_folder,final_image_name)

                        #print(final_image_path)
                        dataset.append({'image_path':final_image_path})

                #print("------")

            else:
                for key in exp_img_param:

                    if ("sn" in key) and (exp_img_param[key] in self.params["cam_sn_list2"]):

                        final_image_name = key.split("_")[0] + ".jpg" # get the correct actual image ## cam_sn_list -> xx_sn -> xx
                        final_image_path = os.path.join(exp_folder,final_image_name)

                        #print(final_image_path)
                        dataset.append({'image_path':final_image_path})

                #print("------")

            if( n % 1000==0 ):

                print(n)


        #print(dataset) #dataset 是一个含有字典的列表 ： dataset = [{image_path:''},{image_path:''}]

        return dataset



    def write_csv(self,dataset_dic,csv_file):
        
        df = pd.DataFrame(dataset_dic)        
        df.to_csv(csv_file)


if(__name__ == '__main__'):

    data = Gen_Dataset(r'V:\Quasar\tools\FaceCapture\src\mirror\config.json')

    dataset = data.get_dataset()

    data.write_csv(dataset,'./src/mirror/dataset/dataset.csv')