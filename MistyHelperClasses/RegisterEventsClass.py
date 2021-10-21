########################  This file defines a class that registers to various MISTY events and returns the live data from Misty
from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.EventFilters import EventFilters
import json
import requests
import time


class RegisterMistyEvents(object):

    def __init__(self,robot_object, event_name, event_type, eventFilter):
        self.robot_object = robot_object
        self.event_name = event_name
        self.event_type = event_type
        self.eventFilter  = eventFilter

    # function that registers to Misty events
    def RegisterForEvents_generic(self):
        self.misty =  self.robot_object
        self.EVENT_NAME = self.event_name
        self.EVENT_TYPE = self.event_type
        self.EVENT_FILTER = self.eventFilter
        self.Data = self.misty.RegisterEvent(self.EVENT_NAME, self.EVENT_TYPE, condition = self.EVENT_FILTER)
        # Wait for the event to get some data before printing the message
        while "just waiting for data" in str(self.Data.data):
            pass
        Msg = self.Data.data["message"]
        return Msg

