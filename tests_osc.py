# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 18:31:28 2025

@author: Damien
"""

from pythonosc.udp_client import SimpleUDPClient

ip = "127.0.0.1"
port = 4000

client = SimpleUDPClient(ip, port)  # Create client

client.send_message("/some/address", 123)   # Send float message
client.send_message("/some/address", [1, 2., "hello"])  # Send message with int, float and string

#%%
from pythonosc import udp_client
import json

client = udp_client.SimpleUDPClient("127.0.0.1", 9000)

data = {"name": "Auralys", "channels": 12, "width": 1.0}
json_str = json.dumps(data)

client.send_message("/mydict", json_str)