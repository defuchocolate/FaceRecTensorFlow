nvidia-docker run -it  -w /FaceRecognition -v /home/$USER/Documents/model:/model -v /home/$USER/Documents/ids:/ids -p 8080:8080 facerecserver
