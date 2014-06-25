#!/usr/bin/env python
from __future__ import division
import roslib; roslib.load_manifest('Kinefly')
import rospy
import rosparam

import copy
import numpy as np

from std_msgs.msg import Header, String

from Kinefly.msg import MsgAnalogIn
from phidgets.srv import SrvPhidgetsInterfaceKitGetAI
from setdict import SetDict


###############################################################################
###############################################################################
# PublishVoltages()
#
# Reads the analog input channels on the Phidgets InterfaceKit, and publish
# the voltages on the topic 'ai'.  The list of channels is specified by the
# parameter 'channels_ai'.
#
class PublishVoltages:

    def __init__(self):
        self.bInitialized = False
        
        # initialize
        self.name = 'publish_voltages'
        rospy.init_node(self.name, anonymous=False)
        self.nodename = rospy.get_name()
        self.namespace = rospy.get_namespace()
        
        # Load the parameters.
        self.params = rospy.get_param('%s' % self.nodename.rstrip('/'), {})
        self.defaults = {'channels_ai':[0,1,2,3,4,5,6,7]}
        SetDict().set_dict_with_preserve(self.params, self.defaults)
        rospy.set_param('%s' % self.nodename.rstrip('/'), self.params)
        
        self.command = None

        # Messages & Services.
        self.stAI = '%s/ai' % self.namespace.rstrip('/')
        self.pubAI       = rospy.Publisher(self.stAI, MsgAnalogIn)
        self.subCommand  = rospy.Subscriber('%s/command' % self.nodename.rstrip('/'), String, self.command_callback, queue_size=1000)
        
        self.get_ai = self.get_wingdata_right = rospy.ServiceProxy('get_ai', SrvPhidgetsInterfaceKitGetAI)
        
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
            rospy.logwarn('Reads the analog input channels on the Phidgets InterfaceKit, and publishes')
            rospy.logwarn('the voltages on the topic ''ai''.  The list of channels is specified by the ')
            rospy.logwarn('parameter ''channels_ai''.')
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
        iCount = 0
        while (not rospy.is_shutdown()):
            header = Header(seq=iCount, stamp=rospy.Time.now())
            try:
                voltages = self.get_ai(self.params['channels_ai'])
            except rospy.service.ServiceException, e:
                self.get_ai = rospy.ServiceProxy(self.stAI, SrvPhidgetsInterfaceKitGetAI)
                
            self.pubAI.publish(header, voltages.voltages)
            iCount += 1



if __name__ == '__main__':

    main = PublishVoltages()
    main.run()

