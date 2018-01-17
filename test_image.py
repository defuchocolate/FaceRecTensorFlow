import sys
import requests

def post_image(img_file):
    """ post image and return the response """
    img = open(img_file, 'rb').read()
    response = requests.post('http://%s:8080/identify' % sys.argv[2], data=img)
    print response.text

if __name__ == '__main__':
	post_image(sys.argv[1])