#!/usr/bin/env python
from __future__ import division
import rospy
import rosparam

import copy
import numpy as np

from std_msgs.msg import Header, String

from Kinefly.msg import MsgAnalogIn, MsgDigitalIn
from phidgets.srv import SrvPhidgetsInterfaceKitGetAI, SrvPhidgetsInterfaceKitGetDI
from setdict import SetDict


###############################################################################
###############################################################################
# AcquireVoltages2Msg()
#
# Read the input voltages on the Phidgets InterfaceKit, and publish
# the values on the topics 'ai' and 'di'.  The lists of channels are 
# specified by the parameters 'channels_ai', 'channels_di'.
#
class AcquireVoltages2Msg:

    def __init__(self):
        self.bInitialized = False
        
        # initialize
        self.name = 'acquirevoltages2msg'
        rospy.init_node(self.name, anonymous=False)
        self.nodename = rospy.get_name()
        self.namespace = rospy.get_namespace()
        
        # Load the parameters.
        self.params = rospy.get_param('%s' % self.nodename.rstrip('/'), {})
        self.defaults = {'channels_ai':[0,1,2,3,4,5,6,7],
                         'channels_di':[0,1,2,3,4,5,6,7]}
        SetDict().set_dict_with_preserve(self.params, self.defaults)
        rospy.set_param('%s' % self.nodename.rstrip('/'), self.params)
        
        self.command = None

        # Messages & Services.
        self.topicAI = '%s/ai' % self.namespace.rstrip('/')
        self.pubAI       = rospy.Publisher(self.topicAI, MsgAnalogIn)
        self.topicDI = '%s/di' % self.namespace.rstrip('/')
        self.pubDI       = rospy.Publisher(self.topicDI, MsgDigitalIn)
        self.subCommand  = rospy.Subscriber('%s/command' % self.nodename.rstrip('/'), String, self.command_callback, queue_size=1000)
        
        self.get_ai = rospy.ServiceProxy('get_ai', SrvPhidgetsInterfaceKitGetAI)
        self.get_di = rospy.ServiceProxy('get_di', SrvPhidgetsInterfaceKitGetDI)
        
        rospy.sleep(1) # Allow time to connect.
        
        self.bInitialized = True
        
        
    # command_callback()
    # Execute any commands sent over the command topic.
    #
    def command_callback(self, msg):
        self.command = msg.data
        
        if (self.command == 'exit'):
            rospy.signal_shutdown('User requested exit.')


        if (self.command == 'help'):
            rospy.logwarn('')
            rospy.logwarn('Reads the input voltages on the Phidgets InterfaceKit, and publishes')
            rospy.logwarn('the values on the topics ''ai'' and ''di''.  The list of channels is specified by the ')
            rospy.logwarn('parameters ''channels_ai'',  ''channels_di''.')
            rospy.logwarn('')
            rospy.logwarn('The %s/command topic accepts the following string commands:' % self.nodename.rstrip('/'))
            rospy.logwarn('  help                 This message.')
            rospy.logwarn('  exit                 Exit the program.')
            rospy.logwarn('')
            rospy.logwarn('You can send the above commands at the shell prompt via:')
            rospy.logwarn('rostopic pub -1 %s/command std_msgs/String commandtext' % self.nodename.rstrip('/'))
            rospy.logwarn('')
            rospy.logwarn('')

    
        
    def run(self):
        channels_ai = self.params['channels_ai']
        channels_di = self.params['channels_di']
        iCount = 0
        while (not rospy.is_shutdown()):
            header = Header(seq=iCount, stamp=rospy.Time.now())
            if (len(channels_ai)>0):
                try:
                    resp = self.get_ai(channels_ai)
                except rospy.service.ServiceException, e:
                    self.get_ai = rospy.ServiceProxy(self.topicAI, SrvPhidgetsInterfaceKitGetAI)
                else:
                    self.pubAI.publish(header, channels_ai, resp.voltages)

            if (len(channels_di)>0):
                try:
                    resp = self.get_di(channels_di)
                except rospy.service.ServiceException, e:
                    self.get_di = rospy.ServiceProxy(self.topicDI, SrvPhidgetsInterfaceKitGetDI)
                else:    
                    self.pubDI.publish(header, channels_di, resp.values)

            iCount += 1



if __name__ == '__main__':

    main = AcquireVoltages2Msg()
    main.run()

