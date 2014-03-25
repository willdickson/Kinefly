#!/usr/bin/env python
from __future__ import division
import roslib; roslib.load_manifest('StrokelitudeROS')
import rospy
import rosparam

import copy
import numpy as np

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
        self.defaults = {'method': 'voltage', # 'voltage' or 'usb';        How we communicate with the panel controller.
                         'pattern_id': 1,
                         'mode': 'velocity',  # 'velocity' or 'position';  Fly is controlling vel or pos.
                         'axis': 'x',         # 'x' or 'y';               The axis on which the frames move.
                         'coeff_voltage':{
                             'adc0':1,  # When using voltage method, coefficients adc0-3 and funcx,y determine how the panels controller interprets its input voltage(s).
                             'adc1':0,  # e.g. xvel = adc0*bnc0 + adc1*bnc1 + adc2*bnc2 + adc3*bnc3 + funcx*f(x) + funcy*f(y); valid on [-128,+127], and 10 corresponds to 1.0.
                             'adc2':0,
                             'adc3':0,
                             'funcx':0,
                             'funcy':0,
                             },
                         'coeff_usb':{  # When using usb method, coefficients x0,xl1,...,yaa,yar determine the pos or vel command sent to the controller over USB.
                             'x0': 0.0,
                             'xl1':1.0,
                             'xl2':0.0,
                             'xr1':-1.0,
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
                         }
        self.set_dict_with_preserve(self.params, self.defaults)
        self.update_coefficients_from_params()
        rospy.set_param('strokelitude/ledpanels', self.params)
        
        self.msgpanels = MsgPanelsCommand(command='all_off', arg1=0, arg2=0, arg3=0, arg4=0, arg5=0, arg6=0)
        

        # Publishers.
        self.pubPanelsCommand = rospy.Publisher('ledpanels/command', MsgPanelsCommand)
        
        # Subscriptions.        
        self.subFlystate = rospy.Subscriber('strokelitude/flystate', MsgFlystate, self.flystate_callback)
        self.subCommand  = rospy.Subscriber('strokelitude2ledpanels/command', String, self.command_callback)
        rospy.sleep(1) # Time to connect publishers & subscribers.


        self.pubPanelsCommand.publish(MsgPanelsCommand(command='set_posfunc_id',  arg1=1,                         arg2=0, arg3=0, arg4=0, arg5=0, arg6=0)) # Set default function ch1.
        self.pubPanelsCommand.publish(MsgPanelsCommand(command='set_posfunc_id',  arg1=2,                         arg2=0, arg3=0, arg4=0, arg5=0, arg6=0)) # Set default function ch2.
        self.pubPanelsCommand.publish(MsgPanelsCommand(command='set_pattern_id',  arg1=self.params['pattern_id'], arg2=0, arg3=0, arg4=0, arg5=0, arg6=0))
        self.pubPanelsCommand.publish(MsgPanelsCommand(command='set_mode',        arg1=0,                         arg2=0, arg3=0, arg4=0, arg5=0, arg6=0)) # xvel=funcx, yvel=funcy
        self.pubPanelsCommand.publish(MsgPanelsCommand(command='set_position',    arg1=0,                         arg2=0, arg3=0, arg4=0, arg5=0, arg6=0)) # Set position to 0
        self.pubPanelsCommand.publish(MsgPanelsCommand(command='send_gain_bias',  arg1=0,                         arg2=0, arg3=0, arg4=0, arg5=0, arg6=0)) # Set vel to 0
        self.pubPanelsCommand.publish(MsgPanelsCommand(command='stop',            arg1=0,                         arg2=0, arg3=0, arg4=0, arg5=0, arg6=0))
        self.pubPanelsCommand.publish(MsgPanelsCommand(command='all_off',         arg1=0,                         arg2=0, arg3=0, arg4=0, arg5=0, arg6=0))

        if (self.params['method']=='voltage'):
            # Assemble a command:  set_mode_(pos|vel)_custom_(x|y) 
            cmd = 'set_mode'
            if (self.params['mode']=='velocity'):
                cmd += '_vel'
            elif (self.params['mode']=='position'):
                cmd += '_pos'
            else:
                rospy.logwarn('strokelitude2ledpanels: mode must be ''velocity'' or ''position''.')
            
            if (self.params['axis']=='x'):
                cmd += '_custom_x'
            elif (self.params['axis']=='y'):
                cmd += '_custom_y'
            else:
                rospy.logwarn('strokelitude2ledpanels: axis must be ''x'' or ''y''.')
            
            # Set the panels controller to the custom mode, with the specified coefficients.
            self.pubPanelsCommand.publish(MsgPanelsCommand(command=cmd, arg1=self.params['coeff_voltage']['adc0'], 
                                                                        arg2=self.params['coeff_voltage']['adc1'], 
                                                                        arg3=self.params['coeff_voltage']['adc2'], 
                                                                        arg4=self.params['coeff_voltage']['adc3'], 
                                                                        arg5=self.params['coeff_voltage']['funcx'], 
                                                                        arg6=self.params['coeff_voltage']['funcy']))
        
        
        self.bInitialized = True
        

    # update_coefficients_from_params()
    #
    # Make a coefficients matrix out of the params dict values.
    # There are two output channels (x,y), and each channel has 
    # coefficients to make a user-specified setting from wing, head, and abdomen angles.
    # 
    def update_coefficients_from_params(self):
        self.a = np.array([[self.params['coeff_usb']['x0'], 
                            self.params['coeff_usb']['xl1'], self.params['coeff_usb']['xl2'], 
                            self.params['coeff_usb']['xr1'], self.params['coeff_usb']['xr2'], 
                            self.params['coeff_usb']['xha'], self.params['coeff_usb']['xhr'], 
                            self.params['coeff_usb']['xaa'], self.params['coeff_usb']['xar']],
                           [self.params['coeff_usb']['y0'], 
                            self.params['coeff_usb']['yl1'], self.params['coeff_usb']['yl2'], 
                            self.params['coeff_usb']['yr1'], self.params['coeff_usb']['yr2'], 
                            self.params['coeff_usb']['yha'], self.params['coeff_usb']['yhr'], 
                            self.params['coeff_usb']['yaa'], self.params['coeff_usb']['yar']]
                          ],
                          dtype=np.float32
                          )
            
            
    def flystate_callback(self, flystate):
        if (self.bRunning):
            self.params = rospy.get_param('strokelitude/ledpanels', {})
            self.set_dict_with_preserve(self.params, self.defaults)
            self.update_coefficients_from_params()
            
            if (self.params['method']=='usb'):
                if (self.params['mode']=='velocity'):
                    msgVel = self.create_msgpanels_vel(flystate)
                    self.pubPanelsCommand.publish(msgVel)
                    #rospy.logwarn('vel: %s' % msgVel)
                else:
                    msgPos = self.create_msgpanels_pos(flystate)
                    self.pubPanelsCommand.publish(msgPos)
                    #rospy.logwarn('pos: %s' % msgPos)
                    
            elif (self.params['method']=='voltage'):
                pass
                
            else:
                rospy.logwarn('strokelitude2ledpanels: method must be ''usb'' or ''voltage''')
    
    
    # create_msgpanels_pos()
    # Return a message to set the panels position.
    #
    def create_msgpanels_pos(self, flystate):
        state = np.array([1.0,
                          flystate.left.angle1,
                          flystate.left.angle2,
                          flystate.right.angle1,
                          flystate.right.angle2,
                          flystate.head.angle,
                          flystate.head.radius,
                          flystate.abdomen.angle,
                          flystate.abdomen.radius
                          ], dtype=np.float32)
        
        pos = np.dot(self.a, state)

        # index is in frames.        
        index_x = int(pos[0])
        index_y = int(pos[1])
        
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
        state = np.array([1.0,
                          flystate.left.angle1,
                          flystate.left.angle2,
                          flystate.right.angle1,
                          flystate.right.angle2,
                          flystate.head.angle,
                          flystate.head.radius,
                          flystate.abdomen.angle,
                          flystate.abdomen.radius
                          ], dtype=np.float32)
        
        vel = np.dot(self.a, state)

        # gain, bias are in frames per sec, times ten, i.e. 10=1.0, 127=12.7 
        gain_x = (int(vel[0]) + 128) % 256 - 128
        bias_x = 0
        gain_y = (int(vel[1]) + 128) % 256 - 128 # As if y were on a sphere, not a cylinder.
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
            self.pubPanelsCommand.publish(MsgPanelsCommand(command='all_off',   arg1=0, arg2=0, arg3=0, arg4=0, arg5=0, arg6=0))
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
        self.pubPanelsCommand.publish(MsgPanelsCommand(command='all_off',   arg1=0, arg2=0, arg3=0, arg4=0, arg5=0, arg6=0))


if __name__ == '__main__':

    s2l = Strokelitude2Ledpanels()
    s2l.run()
