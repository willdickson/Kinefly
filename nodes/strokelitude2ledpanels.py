#!/usr/bin/env python
from __future__ import division
import roslib; roslib.load_manifest('StrokelitudeROS')
import rospy
import rosparam

import copy
import numpy as np
import serial

from std_msgs.msg import Float32, Header, String

from ledpanels.msg import MsgPanelsCommand
from StrokelitudeROS.msg import MsgFlystate, MsgWing, MsgBodypart


###############################################################################
###############################################################################
class Strokelitude2Ledpanels:

    def __init__(self):
        self.bInitialized = False
        self.bRunning = False

        # initialize
        rospy.init_node('strokelitude2ledpanels', anonymous=True)
        
        # Load the parameters.
        self.params = rospy.get_param('strokelitude/ledpanels', {})
        self.defaults = {'x_panels': 24,
                        'y_panels': 3,
                        'mode': 'velocity',
                        'pattern_id': 4,
                        'x0': 0.0,
                        'xl1':2.0,
                        'xl2':0.0,
                        'xr1':-2.0,
                        'xr2':0.0,
                        'xha':0.0,
                        'xhr':0.0,
                        'xaa':0.0,
                        'xar':0.0,
                        'y0': 0.0,
                        'yl1':0.0,
                        'yl2':0.0,
                        'yr1':0.0,
                        'yr2':0.0,
                        'yha':0.0,
                        'yhr':0.0,
                        'yaa':0.0,
                        'yar':0.0,
                        }
        self.set_dict_with_preserve(self.params, self.defaults)
        self.update_coefficients_from_params()
        #rospy.set_param('strokelitude',self.params)
        
        self.msgpanels = MsgPanelsCommand(command='all_off', arg1=0, arg2=0, arg3=0, arg4=0, arg5=0, arg6=0)
        

        # Publishers.
        self.pubPanelsCommand = rospy.Publisher('ledpanels/command', MsgPanelsCommand)
        
        # Subscriptions.        
        self.subFlystate = rospy.Subscriber('strokelitude/flystate', MsgFlystate, self.flystate_callback)
        self.subCommand  = rospy.Subscriber('strokelitude2ledpanels/command', String, self.command_callback)
        rospy.sleep(1) # Time to connect publishers & subscribers.


        self.pubPanelsCommand.publish(MsgPanelsCommand(command='set_posfunc_id',  arg1=1,                         arg2=0, arg3=0, arg4=0, arg5=0, arg6=0)) # Set default function.
        self.pubPanelsCommand.publish(MsgPanelsCommand(command='set_posfunc_id',  arg1=2,                         arg2=0, arg3=0, arg4=0, arg5=0, arg6=0)) # Set default function.
        self.pubPanelsCommand.publish(MsgPanelsCommand(command='set_pattern_id',  arg1=self.params['pattern_id'], arg2=0, arg3=0, arg4=0, arg5=0, arg6=0))
        self.pubPanelsCommand.publish(MsgPanelsCommand(command='set_mode',        arg1=0,                         arg2=0, arg3=0, arg4=0, arg5=0, arg6=0)) # xvel=funcx, yvel=funcy
        self.pubPanelsCommand.publish(MsgPanelsCommand(command='set_position',    arg1=0,                         arg2=0, arg3=0, arg4=0, arg5=0, arg6=0)) # Set position to 0
        self.pubPanelsCommand.publish(MsgPanelsCommand(command='send_gain_bias',  arg1=0,                         arg2=0, arg3=0, arg4=0, arg5=0, arg6=0)) # Set vel to 0
        self.pubPanelsCommand.publish(MsgPanelsCommand(command='stop',            arg1=0,                         arg2=0, arg3=0, arg4=0, arg5=0, arg6=0))
        self.pubPanelsCommand.publish(MsgPanelsCommand(command='all_off',         arg1=0,                         arg2=0, arg3=0, arg4=0, arg5=0, arg6=0))
        self.bInitialized = True
        

    def update_coefficients_from_params(self):
        self.x0  = self.params['x0']
        self.xl1 = self.params['xl1']
        self.xl2 = self.params['xl2']
        self.xr1 = self.params['xr1']
        self.xr2 = self.params['xr2']
        self.xha = self.params['xha']
        self.xhr = self.params['xhr']
        self.xaa = self.params['xaa']
        self.xar = self.params['xar']
        
        self.y0  = self.params['y0']
        self.yl1 = self.params['yl1']
        self.yl2 = self.params['yl2']
        self.yr1 = self.params['yr1']
        self.yr2 = self.params['yr2']
        self.yha = self.params['yha']
        self.yhr = self.params['yhr']
        self.yaa = self.params['yaa']
        self.yar = self.params['yar']
        
        
    def flystate_callback(self, flystate):
        if (self.bRunning):
            #self.params = rospy.get_param('strokelitude/ledpanels', {})
            #self.set_dict_with_preserve(self.params, self.defaults)
            #self.update_coefficients_from_params()
            
            if (self.params['mode']=='velocity'):
                msgVel = self.create_msgpanels_vel(flystate)
                self.pubPanelsCommand.publish(msgVel)
                #rospy.logwarn('vel: %s' % msgVel)
            else:
                msgPos = self.create_msgpanels_pos(flystate)
                self.pubPanelsCommand.publish(msgPos)
                #rospy.logwarn('pos: %s' % msgPos)
    
    
    # create_msgpanels_pos()
    # Return a message to set the panels position.
    #
    def create_msgpanels_pos(self, flystate):
        L1 = flystate.left.angle1
        L2 = flystate.left.angle2
        R1 = flystate.right.angle1
        R2 = flystate.right.angle2
        HA = flystate.head.angle
        HR = flystate.head.radius
        AA = flystate.abdomen.angle
        AR = flystate.abdomen.radius
        
        # L1,L2,R1,R2,HA,HR,AA,AR are all in radians.
        # x0,xl1,xl2,xr1,xr2,xha,xhr,xaa,xar are coefficients to convert to frames.
        xpos = self.x0 + self.xl1*L1 + self.xl2*L2 + \
                         self.xr1*R1 + self.xr2*R2 + \
                         self.xha*HA + self.xhr*HR + \
                         self.xaa*AA + self.xar*AR # Angle + Radius
        ypos = self.y0 + self.yl1*L1 + self.yl2*L2 + \
                         self.yr1*R1 + self.yr2*R2 + \
                         self.yha*HA + self.yhr*HR + \
                         self.yaa*AA + self.yar*AR # Angle + Radius

        # index is in frames.        
        index_x = int(xpos)
        index_y = int(ypos)
        
        msgPos = MsgPanelsCommand(command='set_position', 
                                  arg1=index_x, 
                                  arg2=index_y, 
                                  arg3=0, 
                                  arg4=0, 
                                  arg5=0, 
                                  arg6=0)
        
        return msgPos
    
        
    # create_msgpanels_vel()
    # Return a message to set the panels velocity.
    #
    def create_msgpanels_vel(self, flystate):
        L1 = flystate.left.angle1
        L2 = flystate.left.angle2
        R1 = flystate.right.angle1
        R2 = flystate.right.angle2
        HA = flystate.head.angle
        HR = flystate.head.radius
        AA = flystate.abdomen.angle
        AR = flystate.abdomen.radius
        
        # L1,L2,R1,R2,HA,HR,AA,AR are all in radians.
        # x0,xl1,xl2,xr1,xr2,xha,xhr,xaa,xar are coefficients to convert to frames per sec.
        xvel = self.x0 + self.xl1*L1 + self.xl2*L2 + \
                         self.xr1*R1 + self.xr2*R2 + \
                         self.xha*HA + self.xhr*HR + \
                         self.xaa*AA + self.xar*AR 
        yvel = self.y0 + self.yl1*L1 + self.yl2*L2 + \
                         self.yr1*R1 + self.yr2*R2 + \
                         self.yha*HA + self.yhr*HR + \
                         self.yaa*AA + self.yar*AR

        # gain, bias are in frames per sec, times ten, i.e. 10=1.0, 127=12.7 
        gain_x = (int(xvel) + 128) % 256 - 128
        bias_x = 0
        gain_y = (int(yvel) + 128) % 256 - 128 # As if y were on a sphere, not a cylinder.
        bias_y = 0
        
        msgVel = MsgPanelsCommand(command='send_gain_bias', 
                                  arg1=gain_x, 
                                  arg2=bias_x, 
                                  arg3=gain_y, 
                                  arg4=bias_y, 
                                  arg5=0, 
                                  arg6=0)
        
        return msgVel
    
        
        
    # command_callback()
    # Execute any commands sent over the command topic.
    #
    def command_callback(self, command):
        self.command = command.data
        
        if (self.command == 'exit'):
            self.pubPanelsCommand.publish(MsgPanelsCommand(command='stop',      arg1=0, arg2=0, arg3=0, arg4=0, arg5=0, arg6=0))
            self.bRunning = False
            rospy.signal_shutdown('User requested exit.')


        if (self.command == 'stop'):
            self.pubPanelsCommand.publish(MsgPanelsCommand(command='stop',      arg1=0, arg2=0, arg3=0, arg4=0, arg5=0, arg6=0))
            self.bRunning = False
            
        
        if (self.command == 'start'):
            self.pubPanelsCommand.publish(MsgPanelsCommand(command='set_pattern_id',  arg1=self.params['pattern_id'], arg2=0, arg3=0, arg4=0, arg5=0, arg6=0))
            self.pubPanelsCommand.publish(MsgPanelsCommand(command='start',           arg1=0,                         arg2=0, arg3=0, arg4=0, arg5=0, arg6=0))
            self.bRunning = True
            
        
        if (self.command == 'help'):
            rospy.logwarn('The strokelitude2ledpanels/command topic accepts the following string commands:')
            rospy.logwarn('  help                 This message.')
            rospy.logwarn('  stop                 Stop the ledpanels.')
            rospy.logwarn('  start                Start the ledpanels.')
            rospy.logwarn('  exit                 Exit the program.')
            rospy.logwarn('')
            rospy.logwarn('You can send the above commands at the shell prompt via:')
            rospy.logwarn('rostopic pub -1 strokelitude2ledpanels/command std_msgs/String commandtext')
            rospy.logwarn('')
            rospy.logwarn('Parameters are settable as launch-time parameters.')
            rospy.logwarn('')

    
        
    # set_dict(self, dTarget, dSource, bPreserve)
    # Takes a target dictionary, and enters values from the source dictionary, overwriting or not, as asked.
    # For example,
    #    dT={'a':1, 'b':2}
    #    dS={'a':0, 'c':0}
    #    Set(dT, dS, True)
    #    dT is {'a':1, 'b':2, 'c':0}
    #
    #    dT={'a':1, 'b':2}
    #    dS={'a':0, 'c':0}
    #    Set(dT, dS, False)
    #    dT is {'a':0, 'b':2, 'c':0}
    #
    def set_dict(self, dTarget, dSource, bPreserve):
        for k,v in dSource.iteritems():
            bKeyExists = (k in dTarget)
            if (not bKeyExists) and type(v)==type({}):
                dTarget[k] = {}
            if ((not bKeyExists) or not bPreserve) and (type(v)!=type({})):
                dTarget[k] = v
                    
            if type(v)==type({}):
                self.set_dict(dTarget[k], v, bPreserve)
    
    
    def set_dict_with_preserve(self, dTarget, dSource):
        self.set_dict(dTarget, dSource, True)
    
    def set_dict_with_overwrite(self, dTarget, dSource):
        self.set_dict(dTarget, dSource, False)


    def run(self):
        rospy.spin()


if __name__ == '__main__':

    s2l = Strokelitude2Ledpanels()
    s2l.run()

