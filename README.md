We gave Narrative Vision name to this project. It is an AI-driven project that enables real-time recognition of sign language gestures captured through a webcam. By processing gestures into text using a custom-built dataset of letters (A-Z) and words, the system generates coherent sentences. These sentences are then converted into realistic videos using the pretrained model ALI-ViLab/text-to-video-ms-1.7b. The entire process is integrated into a user-friendly Streamlit-based web platform, offering functionalities like starting/stopping the webcam, clearing outputs, generating and viewing the final video. This innovative solution bridges the gap between sign language communication and digital accessibility, providing a seamless end-to-end experience for users.

Requirements :
1. Jupyter Notebook and either command line or any Python IDE.
2. Minimum 4 GB RAM

Steps :
1. You can download .zip file of project from URL( https://github.com/Vidita08/Sign-Text-Video.git )
2. Upload folder on Jupyter notebook.
3. At first run Collect_Data.ipynb file for collect your own sign gesture data.
   For collecting data you need to show hand gesture for that particular sign infront of webcam\camera and press (q) to captute gesture. 
   Gestures are like : 
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
    22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Road', 27: 'Milk',
    28: 'Ball', 29: 'Football', 30: 'lamp', 31: 'Chair',
    32: 'Book', 33: 'Orange', 34: 'Glass', 35: 'Dog',
    36: 'Cat', 37: 'space', 38: 'Water', 39: 'Man', 40: 'Woman', 41: 'Girl', 42: 'Boy'

   You can modify labels according to your need.
   By running this file and collecting data, there is separate folder will be created i.e .\data. In this folder your data will be stored in .jpg format.
4. After that run Hand_Marks.ipynb file and run code. It detects landmark.
5. Run Train_Mode.ipynb file there are separate 2 codes. Run both code for training and Testing.
6. There will be data.pickle and model.pb file is been created.
7. Run Real_Time.ipynb and check whether system predicts gestures correctly.
8. open test1.py file and update path of model.pb
9. run test1.py on either command line or on any python IDE (VS Code, PyCharm,...etc). 
	streamlit run test1.py


Working of project:
1. At first collect data of your own. 
2. Plot landmarks of collected data. It only landmark hand gesture not facial expression and it only take single hand. It perform annotation for selecting hand gesture.
3. Train model using Random Forest Classifier. Basically it classifies data into Training and Testing. Training contains 80% data and Testing contains 20% data. After that in Training phase it uses Support Vector Machine to classify data in x_train and y_train.
4. data.pickle file contains gestures and there corresponding lables. But it not in human readale format. It is machine readable format.
5. model.pb is a TensorFlow created file format. Basically it used for saving and deploying tensorflow models.
6. For interface purpose we used streamlit web-based application. 
