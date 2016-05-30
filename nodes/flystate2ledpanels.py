#!/usr/bin/env python
from __future__ import division
import rospy
import rosparam
import copy
import numpy as np
from std_msgs.msg import Float32, Header, String
from ledpanels.msg import MsgPanelsCommand
from Kinefly.msg import MsgFlystate
from setdict import SetDict


###############################################################################
###############################################################################
class Flystate2Ledpanels:

    def __init__(self):
        self.bInitialized = False
        self.bRunning = False

        # initialize
        self.name = 'Flystate2ledpanels'
        rospy.init_node(self.name, anonymous=True)
        self.nodename = rospy.get_name()
        self.namespace = rospy.get_namespace()
        
        # Load the parameters.
        self.params = rospy.get_param('%s' % self.nodename.rstrip('/'), {})
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
                         'coeff_usb':{  # When using usb method, coefficients x0,xl1,xl2, ... ,yaa,yar,yxi determine the pos or vel command sent to the controller over USB.
                             'x0': 0.0,
                             'xl1':1.0,
                             'xl2':0.0,
                             'xr1':-1.0,
                             'xr2':0.0,
                             'xha':0.0,
                             'xhr':0.0,
                             'xaa':0.0,
                             'xar':0.0,
                             'xxi':0.0,
                             'y0': 0.0,
                             'yl1':0.0,
                             'yl2':0.0,
                             'yr1':0.0,
                             'yr2':0.0,
                             'yha':0.0,
                             'yhr':0.0,
                             'yaa':0.0,
                             'yar':0.0,
                             'yxi':0.0,
                             }
                         }
        SetDict().set_dict_with_preserve(self.params, self.defaults)
        self.update_coefficients_from_params()
        rospy.set_param('%s' % self.nodename.rstrip('/'), self.params)
        
        self.msgpanels = MsgPanelsCommand(command='all_off', arg1=0, arg2=0, arg3=0, arg4=0, arg5=0, arg6=0)
        

        # Publishers.
        self.pubPanelsCommand = rospy.Publisher('/ledpanels/command', MsgPanelsCommand)
        
        # Subscriptions.        
        self.subFlystate = rospy.Subscriber('%s/flystate' % self.namespace.rstrip('/'), MsgFlystate, self.flystate_callback, queue_size=1000)
        self.subCommand  = rospy.Subscriber('%s/command' % self.nodename.rstrip('/'), String, self.command_callback, queue_size=1000)
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
                rospy.logwarn('%s: mode must be ''velocity'' or ''position''.' % self.name)
            
            if (self.params['axis']=='x'):
                cmd += '_custom_x'
            elif (self.params['axis']=='y'):
                cmd += '_custom_y'
            else:
                rospy.logwarn('%s: axis must be ''x'' or ''y''.' % self.name)
            
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
                            self.params['coeff_usb']['xaa'], self.params['coeff_usb']['xar'],
                            self.params['coeff_usb']['xxi']],
                           [self.params['coeff_usb']['y0'], 
                            self.params['coeff_usb']['yl1'], self.params['coeff_usb']['yl2'], 
                            self.params['coeff_usb']['yr1'], self.params['coeff_usb']['yr2'], 
                            self.params['coeff_usb']['yha'], self.params['coeff_usb']['yhr'], 
                            self.params['coeff_usb']['yaa'], self.params['coeff_usb']['yar'],
                            self.params['coeff_usb']['yxi']]
                          ],
                          dtype=np.float32
                          )
            
            
    def flystate_callback(self, flystate):
        if (self.bRunning):
            self.params = rospy.get_param('%s' % self.nodename.rstrip('/'), {})
            SetDict().set_dict_with_preserve(self.params, self.defaults)
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
                rospy.logwarn('%s: method must be ''usb'' or ''voltage''' % self.name)
    
    
    # create_msgpanels_pos()
    # Return a message to set the panels position.
    #
    def create_msgpanels_pos(self, flystate):
        leftMajor = flystate.left.angles[0] if (0<len(flystate.left.angles)) else 0.0
        leftMinor = flystate.left.angles[1] if (1<len(flystate.left.angles)) else 0.0
        rightMajor = flystate.right.angles[0] if (0<len(flystate.right.angles)) else 0.0
        rightMinor = flystate.right.angles[1] if (1<len(flystate.right.angles)) else 0.0

        angleHead     = flystate.head.angles[0] if (0 < len(flystate.head.angles)) else 0.0
        radiusHead    = flystate.head.radii[0] if (0 < len(flystate.head.radii)) else 0.0
        angleAbdomen  = flystate.abdomen.angles[0] if (0 < len(flystate.abdomen.angles)) else 0.0
        radiusAbdomen = flystate.abdomen.radii[0] if (0 < len(flystate.abdomen.radii)) else 0.0
                                                      
        state = np.array([1.0,
                          leftMajor,
                          leftMinor,
                          rightMajor,
                          rightMinor,
                          angleHead,
                          radiusHead,
                          angleAbdomen,
                          radiusAbdomen,
                          flystate.aux.intensity
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
        leftMajor = flystate.left.angles[0] if (0<len(flystate.left.angles)) else 0.0
        leftMinor = flystate.left.angles[1] if (1<len(flystate.left.angles)) else 0.0
        rightMajor = flystate.right.angles[0] if (0<len(flystate.right.angles)) else 0.0
        rightMinor = flystate.right.angles[1] if (1<len(flystate.right.angles)) else 0.0

        angleHead     = flystate.head.angles[0] if (0 < len(flystate.head.angles)) else 0.0
        radiusHead    = flystate.head.radii[0] if (0 < len(flystate.head.radii)) else 0.0
        angleAbdomen  = flystate.abdomen.angles[0] if (0 < len(flystate.abdomen.angles)) else 0.0
        radiusAbdomen = flystate.abdomen.radii[0] if (0 < len(flystate.abdomen.radii)) else 0.0
                                                      
        state = np.array([1.0,
                          leftMajor,
                          leftMinor,
                          rightMajor,
                          rightMinor,
                          angleHead,
                          radiusHead,
                          angleAbdomen,
                          radiusAbdomen,
                          flystate.aux.intensity
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
            rospy.logwarn('The %s/command topic accepts the following string commands:' % self.nodename.rstrip('/'))
            rospy.logwarn('  help                 This message.')
            rospy.logwarn('  stop                 Stop the ledpanels.')
            rospy.logwarn('  start                Start the ledpanels.')
            rospy.logwarn('  exit                 Exit the program.')
            rospy.logwarn('')
            rospy.logwarn('You can send the above commands at the shell prompt via:')
            rospy.logwarn('rostopic pub -1 %s/command std_msgs/String commandtext' % self.nodename.rstrip('/'))
            rospy.logwarn('')
            rospy.logwarn('Parameters are settable as launch-time parameters.')
            rospy.logwarn('')

    
        
    def run(self):
        rospy.spin()
        self.pubPanelsCommand.publish(MsgPanelsCommand(command='all_off',   arg1=0, arg2=0, arg3=0, arg4=0, arg5=0, arg6=0))


if __name__ == '__main__':

    k2l = Flystate2Ledpanels()
    k2l.run()

