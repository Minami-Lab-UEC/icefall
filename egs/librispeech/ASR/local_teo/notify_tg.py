#!/usr/bin/env python3


import requests
import sys
from configparser import ConfigParser

config = ConfigParser()
config.read("misc.ini")
config = config["TELEGRAM"]

token = config["token"] 
chat_id = config["chat_id"] 
url_req = "https://api.telegram.org/bot" + token + "/sendMessage?chat_id=" + chat_id \
    + "&text="

msg = sys.argv[1:]
requests.get(url = (url_req + '\n'.join(msg)))

"""
; # Sample of config file .ini. 
[TELEGRAM]
token = 
chat_id = 

"""
