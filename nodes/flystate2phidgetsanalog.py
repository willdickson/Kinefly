#!/usr/bin/env python
from __future__ import division
import rospy
import rosparam
import copy
import numpy as np
from std_msgs.msg import String
from Kinefly.msg import MsgFlystate
import Phidgets
import Phidgets.Devices.Analog
from setdict import SetDict


###############################################################################
###############################################################################
class Flystate2PhidgetsAnalog:

    def __init__(self):
        self.bInitialized = False
        self.iCount = 0
        self.bAttached = False
        
        # initialize
        self.name = 'Flystate2PhidgetsAnalog'
        rospy.init_node(self.name, anonymous=True)
        self.nodename = rospy.get_name()
        self.namespace = rospy.get_namespace()
        
        # Load the parameters.
        self.params = rospy.get_param('%s' % self.nodename.rstrip('/'), {})
        self.defaults = {'v0enable':True, 'v1enable':True, 'v2enable':True, 'v3enable':True, 
                         'v00': 0.0, 'v0l1':1.0, 'v0l2':0.0, 'v0lr':0.0, 'v0r1':0.0,  'v0r2':0.0, 'v0rr':0.0, 'v0ha':0.0, 'v0hr':0.0, 'v0aa':0.0, 'v0ar':0.0, 'v0xi':0.0, # L
                         'v10': 0.0, 'v1l1':0.0, 'v1l2':0.0, 'v1lr':0.0, 'v1r1':1.0,  'v1r2':0.0, 'v1rr':0.0, 'v1ha':0.0, 'v1hr':0.0, 'v1aa':0.0, 'v1ar':0.0, 'v1xi':0.0, # R
                         'v20': 0.0, 'v2l1':1.0, 'v2l2':0.0, 'v2lr':0.0, 'v2r1':-1.0, 'v2r2':0.0, 'v2rr':0.0, 'v2ha':0.0, 'v2hr':0.0, 'v2aa':0.0, 'v2ar':0.0, 'v2xi':0.0, # L-R
                         'v30': 0.0, 'v3l1':1.0, 'v3l2':0.0, 'v3lr':0.0, 'v3r1':1.0,  'v3r2':0.0, 'v3rr':0.0, 'v3ha':0.0, 'v3hr':0.0, 'v3aa':0.0, 'v3ar':0.0, 'v3xi':0.0, # L+R
                         'autorange':False, 
                         'serial':0         # The serial number of the Phidget.  0==any.
                         }
        SetDict().set_dict_with_preserve(self.params, self.defaults)
        rospy.set_param('%s' % self.nodename.rstrip('/'), self.params)
        

        self.stateMin = np.array([ np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf])
        self.stateMax = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        
        self.update_coefficients()
        
        
        # Connect to the Phidget.
        self.analog = Phidgets.Devices.Analog.Analog()
        if (self.params['serial']==0):
            self.analog.openPhidget()
        else:
            self.analog.openPhidget(self.params['serial'])
            
        self.analog.setOnAttachHandler(self.attach_callback)
        self.analog.setOnDetachHandler(self.detach_callback)

        # Subscriptions.        
        self.subFlystate = rospy.Subscriber('%s/flystate' % self.namespace.rstrip('/'), MsgFlystate, self.flystate_callback, queue_size=1000)
        self.subCommand  = rospy.Subscriber('%s/command' % self.nodename.rstrip('/'), String, self.command_callback, queue_size=1000)
        #rospy.sleep(1) # Allow time to connect publishers & subscribers.

        self.bInitialized = True
        
        
    def attach_callback(self, phidget):
        for i in range(4):
            self.analog.setEnabled(i, self.enable[i])

        self.phidgetserial = self.analog.getSerialNum()
        self.phidgetname = self.analog.getDeviceName()
        rospy.sleep(1) # Wait so that other nodes can display their banner first.
        rospy.logwarn('%s - %s Attached: serial=%s' % (self.namespace, self.phidgetname, self.phidgetserial))
        self.bAttached = True
        

    def detach_callback(self, phidget):
        rospy.logwarn ('%s - %s Detached: serial=%s.' % (self.namespace, self.phidgetname, self.phidgetserial))
        self.bAttached = False


    def update_coefficients(self):
        if (not self.params['autorange']):
            self.update_coefficients_from_params()
        else:
            self.update_coefficients_from_autorange()
            

        self.enable = [self.params['v0enable'], self.params['v1enable'], self.params['v2enable'], self.params['v3enable']]  

    
    # update_coefficients_from_params()
    #
    # Make a coefficients matrix out of the params dict values.
    # There are four voltage channels (v0,v1,v2,v3), and each channel has 
    # coefficients to make a user-specified voltage from wing, head, and abdomen angles.
    # 
    def update_coefficients_from_params(self):
        self.a = np.array([[self.params['v00'], self.params['v0l1'], self.params['v0l2'], self.params['v0lr'], self.params['v0r1'], self.params['v0r2'], self.params['v0rr'], self.params['v0ha'], self.params['v0hr'], self.params['v0aa'], self.params['v0ar'], self.params['v0xi']],
                           [self.params['v10'], self.params['v1l1'], self.params['v1l2'], self.params['v1lr'], self.params['v1r1'], self.params['v1r2'], self.params['v1rr'], self.params['v1ha'], self.params['v1hr'], self.params['v1aa'], self.params['v1ar'], self.params['v1xi']],
                           [self.params['v20'], self.params['v2l1'], self.params['v2l2'], self.params['v2lr'], self.params['v2r1'], self.params['v2r2'], self.params['v2rr'], self.params['v2ha'], self.params['v2hr'], self.params['v2aa'], self.params['v2ar'], self.params['v2xi']],
                           [self.params['v30'], self.params['v3l1'], self.params['v3l2'], self.params['v3lr'], self.params['v3r1'], self.params['v3r2'], self.params['v3rr'], self.params['v3ha'], self.params['v3hr'], self.params['v3aa'], self.params['v3ar'], self.params['v3xi']]
                          ],
                          dtype=np.float32
                          )
            
            
    # update_coefficients_from_autorange()
    #
    # Make a coefficients matrix to automatically set voltage ranges.
    # Set the voltage outputs 0,1,2,3 to L, R, L-R, L+R, respectively, such that they each span the voltage range [-10,+10].
    # 
    def update_coefficients_from_autorange(self):
        lmax = self.stateMax[1]
        lmin = self.stateMin[1]
        lmean = (lmax-lmin)/2

        rmax = self.stateMax[4]
        rmin = self.stateMin[4]
        rmean = (rmax-rmin)/2

        # From Matlab:
        # q0=solve(10==a0+a1*lmax, -10==a0+a1*lmin, a0,a1)
        # q1=solve(10==a0+a3*rmax, -10==a0+a3*rmin, a0,a3)
        # q2=solve(10==a0+a1*lmax+a3*rmin, -10==a0+a1*lmin+a3*rmax, 0==a0+a1*lmin+a3*rmin, a0,a1,a3)
        # q3=solve(10==a0+a1*lmax+a3*rmax, -10==a0+a1*lmin+a3*rmin, 0==a0+a1*lmax+a3*rmin, a0,a1,a3)

        a00 = -(10*(lmax+lmin))/(lmax-lmin)
        a01 = 20/(lmax-lmin)
        
        a10 = -(10*(rmax+rmin))/(rmax-rmin)
        a13 = 20/(rmax-rmin)
        
        a20 = (10*(lmax*rmin - lmin*rmax))/((lmax - lmin)*(rmax - rmin))
        a21 = 10/(lmax - lmin)
        a23 = -10/(rmax - rmin)
        
        a30 = -(10*(lmax*rmax - lmin*rmin))/((lmax - lmin)*(rmax - rmin))
        a31 = 10/(lmax - lmin)
        a33 = 10/(rmax - rmin)
                
        #                     1   L1   L2  LR   R1 R2 RR HA HR AA AR XI
        self.a = np.array([[a00, a01,   0,  0,   0, 0, 0, 0, 0, 0, 0, 0],
                           [a10,   0,   0,  0, a13, 0, 0, 0, 0, 0, 0, 0],
                           [a20, a21,   0,  0, a23, 0, 0, 0, 0, 0, 0, 0],
                           [a30, a31,   0,  0, a33, 0, 0, 0, 0, 0, 0, 0]],
                          )
        
        
    def flystate_callback(self, flystate):
        self.iCount += 1
        
        if (self.bAttached):
            voltages = self.voltages_from_flystate(flystate)

            for i in range(4):
                if (self.enable[i]):
                    try:
                        self.analog.setVoltage(i, voltages[i])
                    except Phidgets.PhidgetException.PhidgetException:
                        pass
            
    
    
    # get_voltages()
    #
    def voltages_from_flystate(self, flystate):
        angle1Left = flystate.left.angles[0] if (0<len(flystate.left.angles)) else 0.0
        angle2Left = flystate.left.angles[1] if (1<len(flystate.left.angles)) else 0.0
        radiusLeft = flystate.left.radii[0] if (0<len(flystate.left.radii)) else 0.0
        angle1Right = flystate.right.angles[0] if (0<len(flystate.right.angles)) else 0.0
        angle2Right = flystate.right.angles[1] if (1<len(flystate.right.angles)) else 0.0
        radiusRight = flystate.right.radii[0] if (0<len(flystate.right.radii)) else 0.0
            
        angleHead     = flystate.head.angles[0] if (0 < len(flystate.head.angles)) else 0.0
        radiusHead    = flystate.head.radii[0] if (0 < len(flystate.head.radii)) else 0.0
        angleAbdomen  = flystate.abdomen.angles[0] if (0 < len(flystate.abdomen.angles)) else 0.0
        radiusAbdomen = flystate.abdomen.radii[0] if (0 < len(flystate.abdomen.radii)) else 0.0
                               
        state = np.array([1.0,
                          angle1Left,
                          angle2Left,
                          radiusLeft,
                          angle1Right,
                          angle2Right,
                          radiusRight,
                          angleHead,
                          radiusHead,
                          angleAbdomen,
                          radiusAbdomen,
                          flystate.aux.intensity
                          ], dtype=np.float32)
        
        if (self.params['autorange']):
            if (self.iCount>10):
                self.stateMin = np.min([self.stateMin, state], 0)
                self.stateMax = np.max([state, self.stateMax], 0)
    
                # Decay toward the mean.
                stateMean = (self.stateMin + self.stateMax) * 0.5
                d = (self.stateMax - stateMean) * 0.001
                self.stateMax -= d
                self.stateMin += d
                
            #rospy.logwarn((self.stateMin,self.stateMax))
            self.update_coefficients_from_autorange()
            
        voltages = np.dot(self.a, state)
        
        # L1,L2,R1,R2,HA,AA are all in radians; LR,RR are in pixels; XI is intensity on the range [0,1].
        # v00,v0l1,v0l2,v0lr,v0r1,v0r2,v0rr,v0ha,v0hr,v0aa,v0ar,v0xi are coefficients to convert to voltage.
#         voltages[0] = self.v00 + self.v0l1*L1 + self.v0l2*L2 + self.v0lr*LR + \
#                                  self.v0r1*R1 + self.v0r2*R2 + self.v0rr*RR + \
#                                  self.v0ha*HA + self.v0hr*HR + \
#                                  self.v0aa*AA + self.v0ar*AR + \ # Angle + Radius
#                                  self.v0xi*XI                    # Aux intensity
                       
        return voltages.clip(-10.0, 10.0)

    
        
    # command_callback()
    # Execute any commands sent over the command topic.
    #
    def command_callback(self, command):
        self.command = command.data
        
        if (self.command == 'exit'):
            rospy.signal_shutdown('User requested exit.')


        if (self.command == 'help'):
            rospy.logwarn('The %s/command topic accepts the following string commands:' % self.nodename.rstrip('/'))
            rospy.logwarn('  help                 This message.')
            rospy.logwarn('  exit                 Exit the program.')
            rospy.logwarn('')
            rospy.logwarn('You can send the above commands at the shell prompt via:')
            rospy.logwarn('rostopic pub -1 %s/command std_msgs/String commandtext' % self.nodename.rstrip('/'))
            rospy.logwarn('')
            rospy.logwarn('Parameters are settable as launch-time parameters.')
            rospy.logwarn('')

    
        
    def run(self):
        rospy.spin()

        if (self.analog.isAttached()):
            for i in range(4):
                self.analog.setVoltage(i, 0.0)
        
        self.analog.closePhidget()


if __name__ == '__main__':

    s2pa = Flystate2PhidgetsAnalog()
    s2pa.run()

