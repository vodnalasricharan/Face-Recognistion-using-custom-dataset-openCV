# Face-Recognistion-openCV
## This project is done for Fun,to recognize the faces of me and my friends
### I have done this in two ways
       1)Using vgg16() as recognizer(which didn't give satisfying results)
       2)Using opencv face LBPHFaceRecognizer
## Using vgg16()
        used fies are:
              load_data.py
              model.py
              faces_train.py
              predict.py(for image prediction)
              face_image.py(also for image prediction...just checked if anything other works well)
              video-predict.py(for live video prediction)
I can't provide all the dataset to you as it is custom dataset made by me.😊
<br>Run the codes in the same order as provided✌
<br>To load data using load_data.py the data should be put in a data folder and images in their respective folders of their label name<br>
![Screenshot](ss1.PNG)
<br>Then run faces_train.py
<br>It takes .csv file and directory of the images obtained by load_data.py
<br>If you want you can place all these images obtained by load_data.py in a directory(folder) named train
<br>After sucessfull execution of all the above codes .....fianlly you can run any prediction code(predict.py/face_image.py/video-predict.py)
<h4>The results are not satisfactory as the dataset near me is not balanced.If you want you can try this on your custom dataset😉</h4>
<br>
## Using LBPHFaceRecognizer
              used files are:
                     data_load2.py
                     training.py
                     recognizer_img.py(for image prediction)
                     recognizer_video.py(for video prediction)
Same as above create a data folder with mages in their respective folders of their label name.
<br>Run the codes in the above sequence😃
