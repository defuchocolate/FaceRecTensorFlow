# FaceRecTensorFlow

Example server serving a tensorflow model.

#### To run: 
1. Copy the models folders to /home/$USER/Documents/models
2. Create a folder called /home/$USER/Documents/ids
3. Inside ids add folders with the names of the people you want to identify
4. Add photos containing only the face of the person you want to identify to the corresponding folder
5. `. run.sh` to run docker version or run your own version of the comand to suit your environment

To test:
Upload a photo to the server using the python script:
`python test_image.py ~/Pictures/Webcam/2017-12-23-151854.jpg localhost`
