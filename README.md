# Face-Recognistion-using-custom-dataset-openCV
## This project is done for Fun,to recognize the faces of me and my friends
First download the zip file of repository and extract it
<div align="center">
    <img src="./downloadzip.PNG" width="400px"</img> 
</div>
<br>After that you can see a folder named Face-Recognisition-using-custom-dataset-openCV
<br>Right click on the folder and click on pycharm open as project (you can use your favourite IDE if any)
<div align="center">
    <img src="./openpycharm.PNG" width="400px"</img> 
</div>
<br><h5>Set the project interpreter in virtual environment(Now I can't tell you how to set virtual environment in pycharm🤷‍♀️,you can search in youtube or google it to know how to set virtual environment)</h5> 
Make sure you install all the required libraries mentioned in requirements.txt
<br>&ensp;&ensp;To install open terminal in pycharam and type
<br>&ensp;&ensp;&ensp;&ensp;pip install -r requirements.txt<br>
<div align="center">
    <img src="./terminal.PNG" width="400px"</img> 
</div>
(If you still face any issues after executing any code snippet,kindly install the libraries manually😏 and run again)
<br>In this repository 
 <br>&ensp;&ensp;&ensp;&ensp;faces.py
 <br>&ensp;&ensp;&ensp;&ensp;face_image.py 
<br>are used to detect the faces in the image
<br>I used harcascade facedetection data which is used in these code snippets ...please ensure that you provide the correct path if you changed the files

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
<div align="center">
    <img src="./datafolder.PNG" width="400px"</img> 
</div><br>
<div align="center">
    <img src="./ss1.PNG" width="400px"</img> 
</div>
<br>Then run faces_train.py
<br>It takes .csv file and directory of the images obtained by load_data.py
<br>If you want you can place all these images obtained by load_data.py in a directory(folder) named train
<br>After sucessfull execution of all the above codes .....fianlly you can run any prediction code(predict.py/face_image.py/video-predict.py)
<h4>The results are not satisfactory as the dataset near me is not balanced.If you want you can try this on your custom dataset😉</h4>

## Using LBPHFaceRecognizer
              
              used files are:
                     data_load2.py
                     training.py
                     recognizer_img.py(for image prediction)
                     recognizer_video.py(for video prediction)
<br>Same as above create a data folder with images in their respective folders of their label name.
<br>Run the codes in the above sequence😃
<br>The output looks something like this(It detects other persons as unknown,whose images are not included in training data)<br>
<div align="center">
    <img src="./ss3.PNG" width="400px"</img> 
</div>
<br>If you have more number of images then use this model,It is giving good results if you have more images.But as harcascade doesn't recognize faces which faces sidewards,This is a major drawback.
<br>If you're having private third party xml file,then you can use it.
<h4>This model is giving bit stable and acuurate results for me.Hope this works on your dataset too🤞</h4>


