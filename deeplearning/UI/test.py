from Tkinter import *
import tkFileDialog
from PIL import Image, ImageTk, ImageDraw, ImageFont

from pathlib import Path
import os
import matplotlib

from extract_frame import getKeyFrames
from __init__ import mySVM

matplotlib.use('TkAgg')

import moviepy.editor as mp
import os.path
import glob
import numpy as np
import scipy.io as sio
from video import Video
import cv2
import traceback

from extract_acoustic import getAcousticFeatures
debug = True

class UI_class:
    def __init__(self, master, search_path, frame_storing_path):
        self.search_path = search_path
        self.master = master
        self.frame_storing_path = frame_storing_path
        topframe = Frame(self.master)
        topframe.pack()

        #Buttons
        topspace = Label(topframe).grid(row=0, columnspan=2)
        self.bbutton= Button(topframe, text=" Choose an video ", command=self.browse_query_img)
        self.bbutton.grid(row=1, column=1)
        self.cbutton = Button(topframe, text=" Estimate its venue ", command=self.show_venue_category)
        self.cbutton.grid(row=1, column=2)
        downspace = Label(topframe).grid(row=3, columnspan=4)

        input_path = str(Path(os.getcwd()).parent.joinpath("store/acoustic/input"))
        output_path = str(Path(os.getcwd()).parent.joinpath("store/acoustic/output"))

        #X_train
        training_path = str(Path(os.getcwd()).parent.joinpath("CS2108-Vine-Dataset/vine/training"))
        store_path_train = input_path
        self.X_train = self.extract_x_set(training_path, store_path_train, True, "X_train")

        #X_test
        validation_path = str(Path(os.getcwd()).parent.joinpath("CS2108-Vine-Dataset/vine/validation"))
        store_path_valid = input_path
        self.X_test = self.extract_x_set(validation_path, store_path_valid, True, "X_test")

        #Y_train
        venue_path = str(Path(os.getcwd()).parent.joinpath("CS2108-Vine-Dataset/vine-venue-training.txt"))
        store_path_venue = input_path
        self.Y_train = self.extract_y_set(venue_path, store_path_venue, True, "Y_train")

        #Y_gnd
        validation_venue_path = str(Path(os.getcwd()).parent.joinpath("CS2108-Vine-Dataset/vine-venue-validation.txt"))
        store_path_venue = input_path
        self.Y_gnd = self.extract_y_set(validation_venue_path, store_path_venue, True, "Y_gnd")

        #Testing SVM
        #TODO put Y_train X_train X_test Y_gnd in one same input.mat workplace -- code is commeted out to run without error
        self.Y_predicted = mySVM(input_path, output_path)


        #TODO allow user to select folder path and use self.extract_x_training_set to get X_test
        #TODO allow user to select 1 file path / make new method like self.extract_x_training_set for 1 file


        #TODO use classifier code to get Y_predicted which is an array like this [2.1, 3.4]
        #TODO (basically if it predicts 2.1 = venue 2, 3.6 = venue 4 etc) round to nearest integer to get venue #

        #TODO display venue

        #TODO calculate f1 by trying out every weight possible
        f1 = self.compare_vectors(self.Y_gnd, self.Y_predicted)
        print(f1)
        #TODO display best weight combination

        self.master.mainloop()


    def browse_query_img(self):
        self.query_img_frame = Frame(self.master)
        self.query_img_frame.pack()
        from tkFileDialog import askopenfilename
        self.filename = tkFileDialog.askopenfile(title='Choose an Video File').name

        allframes = os.listdir(self.frame_storing_path)
        self.videoname = self.filename.strip().split("/")[-1].replace(".mp4","")
        video_file = self.filename
        vidcap = cv2.VideoCapture(video_file)
        file_path = "/Users/Admin/CS2108Assignment_2/deeplearning/data/frame/" + self.videoname + "-"
        print(file_path)
        keyframes = getKeyFrames(vidcap=vidcap, store_frame_path=file_path)

        self.frames = []
        for frame in allframes:
            if self.videoname +"-frame" in frame:
                self.frames.append(self.frame_storing_path + frame)

        COLUMNS = len(self.frames)
        self.columns = COLUMNS
        print (COLUMNS)
        image_count = 0

        if COLUMNS == 0:
            self.frames.append("none.png")
            print("Please extract the key frames for the selected video first!!!")
            COLUMNS = 1

        for frame in self.frames:

            r, c = divmod(image_count, COLUMNS)
            try:
                im = Image.open(frame)
                resized = im.resize((100, 100), Image.ANTIALIAS)
                tkimage = ImageTk.PhotoImage(resized)

                myvar = Label(self.query_img_frame, image=tkimage)
                myvar.image = tkimage
                myvar.grid(row=r, column=c)

                image_count += 1
                self.lastR = r
                self.lastC = c
            except Exception, e:
                continue

        try:
            audiopath = os.path.dirname(self.filename)
            audiopath = os.path.dirname(audiopath)
            audiopath = os.path.join(audiopath, "audio/test.mp3")
            self.getAudioClip(self.filename, audiopath)
            feature_mfcc, feature_spect, feature_zerocrossing, feature_energy = getAcousticFeatures(audiopath)

        except Exception, e:
            print(e)

        self.query_img_frame.mainloop()


    def show_venue_category(self):
        if self.columns == 0:
            print("Please extract the key frames for the selected video first!!!")
        else:
            # Please note that, you need to write your own classifier to estimate the venue category to show blow.
            if self.videoname == '1':
               venue_text = "Home"
            elif self.videoname == '2':
                venue_text = 'Bridge'
            elif self.videoname == '4':
                venue_text = 'Park'

            venue_img = Image.open("venue_background.jpg")
            draw = ImageDraw.Draw(venue_img)

            font = ImageFont.truetype("/Library/Fonts/Arial.ttf",size=66)

            draw.text((50,50), venue_text, (0, 0, 0), font=font)

            resized = venue_img.resize((100, 100), Image.ANTIALIAS)
            tkimage =ImageTk.PhotoImage(resized)

            myvar = Label(self.query_img_frame, image=tkimage)
            myvar.image= tkimage
            myvar.grid(row=self.lastR, column=self.lastC+1)

        self.query_img_frame.mainloop()


    def getAudioClip(self, video_reading_path, audio_storing_path):
        clip = mp.VideoFileClip(video_reading_path)
        clip.audio.write_audiofile(audio_storing_path)


    def extract_x_set(self, pathname, store_path, is_storing, name):
        current_dir = os.getcwd()
        if (self.set_exists_or_create(store_path)):
            try:
                dic = sio.loadmat(store_path)
                if (debug):
                    print(dic.keys())
                matrix = dic[name]
                return matrix
            except Exception, err:
                if (debug):
                    traceback.print_exc()
                pass

        os.chdir(pathname)
        videos = []
        column_size = 0
        ctr = 0

        for file in glob.glob("*.mp4"):
            video_path = os.path.abspath(file)
            feature_mfcc, feature_spect, feature_zerocrossing, feature_energy = getAcousticFeatures(video_path)

            # extract feature vector and get the max feature_vector size (they're not all the same size)
            feature_vector = self.combine_features(feature_mfcc, feature_spect, feature_zerocrossing, feature_energy)
            column_size = max(len(feature_vector), column_size)

            video = Video(file, feature_vector)
            videos.append(video)
            ctr += 1

        matrix = self.combine_videos(videos, len(videos), column_size)
        os.chdir(current_dir)
        if (is_storing):
            matrix = np.array(matrix)
            dic = sio.loadmat(store_path)
            dic[name] = matrix
            sio.savemat(store_path, dic)

        return matrix


    def extract_y_set(self, pathname, store_path, is_storing, name):
        if (self.set_exists_or_create(store_path)):
            try:
                dic = sio.loadmat(store_path)
                if (debug):
                    print(dic.keys())
                    arr = dic[name]
                return arr
            except Exception, err:
                if (debug):
                    traceback.print_exc()
                pass

        arr = []
        with open(pathname, 'r') as file:
            lines = file.readlines()
            lines.sort()
            for line in lines:
                string = line.split('\t')[1].strip()
                venue = float(string)
                arr.append(venue)

        if (is_storing):
            arr = np.array([arr]).transpose()
            dic = sio.loadmat(store_path)
            dic[name] = arr
            sio.savemat(store_path, dic)
        return arr


    def combine_features(self, feature_mfcc, feature_spect, feature_zerocrossing, feature_energy):
        feature_mfcc = self.mean_pooling(feature_mfcc)
        feature_spect = self.mean_pooling(feature_spect)
        feature_zerocrossing = self.mean_pooling(feature_zerocrossing)
        feature_energy = self.mean_pooling(feature_energy)
        result = []

        for i in range(len(feature_mfcc)):
            #if (feature_mfcc[i] <0):
                #feature_mfcc[i] = 0
            result.append(feature_mfcc[i])
        for i in range(len(feature_spect)):
            #if (feature_spect[i] < 0):
                #feature_spect[i] = 0
            result.append(feature_spect[i])
        #for i in range(len(feature_zerocrossing)):
            #if (feature_zerocrossing[i] <0):
                #feature_zerocrossing [i] = 0
            #result.append(feature_zerocrossing[i])
        #for i in range(len(feature_energy)):
            #if (feature_energy[i] <0):
                #feature_energy[i] = 0
            #result.append(feature_energy[i])
        return result


    def mean_pooling(self, lst):
        dim = np.ndim(lst)
        if dim == 1:
            return lst
        return np.mean(lst, axis=0)


    def combine_videos(self, videos, rows, cols):
        matrix = np.zeros((rows, cols))
        for row in range(len(videos)):
            vector = videos[row].feature_vector
            #for col in range(len(vector)):
            for col in range(len(vector)):
                matrix[row][col] = vector[col]
        return matrix

    def set_exists_or_create(self, pathname):
        my_file = Path(pathname + ".mat")
        if my_file.is_file():
            return True
        sio.savemat(pathname, {})
        return False

    def compare_vectors(self, vector1, vector2):
        total = len(vector1)
        ctr = 0.0
        bad_values = 0
        number_f = 0
        for row in range(len(vector1)):
            val1 = vector1[row][0]
            val2 = vector2.flat[row]
            ctr += 1 if (val1 == round(val2, 0)) else 0
            bad_values += 1 if (val1 != round(val2, 0)) else 0
            number_f += 1 if (val2 == 15.5) else 0
        f1 = ctr / total
        return f1


root = Tk()
try:
    window = UI_class(root, search_path='../data/video/', frame_storing_path='../data/frame/')
except Exception:
    traceback.print_exc()