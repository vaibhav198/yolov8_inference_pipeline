Steps to run the project/pipeline-

Make sure you are inside the "object_detection_pipeline" folder. open a terminal and follow the ste below - 

1. edit the .env file to set the different parameters (like "MODE" means how we are feeding the data- 'images' or 'video' or 'webcam')
2. Create a virtual environment (python3 -m venv ./venv) 
3. Activate your environment (source ./venv/bin/activate)
4. Now install package from requirements.txt (pip install -r requirements.txt)
5. After installing the package, to start the pipeline run- (python app.py)
6. Thats it. your pipeline is up
---------------------------------------------------------------------------------------------
--------------folder/dir overview------------------------------------------------------------
-Now you can have a look at "output" folder. This folder contains all the images where object detection and classification happened. 
-"missed_images" folder contains all the images where detection not happened.
--------------------------------------------------------------------------------------------
-------------Model overview-----------------------------------------------------------------
I have used YOLOv8 pretained model for object identification. The model is not too much accurate on the video/image data that we are using in here.
In order to get the better performance we need to fine tune it.

