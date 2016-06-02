#!/usr/bin/env python
import rospy
import numpy as np



###############################################################################
###############################################################################
class WingbeatDetector(object):
    def __init__(self, fw_min, fw_max):
        self.n = 64
        self.buffer = np.zeros([2*self.n, 2]) # Holds intensities & framerates.
        self.set(fw_min, fw_max)
        
                
    def set(self, fw_min, fw_max):
        self.i = 0
        
        # Set the desired passband.
        self.fw_min = fw_min
        self.fw_max = fw_max
        self.fw_center = (fw_min + fw_max) / 2.0

        # Framerate needed to measure the desired band.        
        self.fs_dict = self.fs_dict_from_wingband(fw_min, fw_max)
        if (len(self.fs_dict['fs_range_list'])>0):
            (self.fs_lo, self.fs_hi) = self.fs_dict['fs_range_list'][0]
        else:
            (self.fs_lo, self.fs_hi) = (0.0, 0.0)
        

    def warn(self):    
        rospy.logwarn('Note: The wingbeat detector is set to measure wingbeat frequencies in the ')
        rospy.logwarn('band [%0.1f, %0.1f] Hz.  To make a valid measurement, the camera ' % (self.fw_min, self.fw_max))
        rospy.logwarn('framerate must be in, and stay in, one of the following ranges:')
        rospy.logwarn(self.fs_dict['fs_range_list'])


    # fs_dict_from_wingband()
    # Compute the camera frequency range [fs_lo, fs_hi] required to undersample the given wingbeat frequency band.
    # fw_min:    Lower bound for wingbeat frequency.
    # fw_max:    Upper bound for wingbeat frequency.
    # fs_lo:   Lower bound for sampling frequency.
    # fs_hi:   Upper bound for sampling frequency.
    #
    # Returns a list of all the possible framerate ranges: [[lo,hi],[lo,hi],...]
    #
    def fs_dict_from_wingband(self, fw_min, fw_max):
        fs_dict = {}
        fs_range_list = []
        m_list = []
        
        bw = fw_max - fw_min
        m = 1
        while (True):
            fs_hi =     (2.0 * self.fw_center - bw) / m
            fs_lo = max((2.0 * self.fw_center + bw) / (m+1), 2*bw)
            if (2*bw < fs_hi):
                fs_range_list.append([fs_lo, fs_hi])
                m_list.append(m)
            else:
                break
            m += 1
        
        # Put the list into low-to-high order.
        fs_range_list.reverse()
        m_list.reverse()

        fs_dict['fs_range_list'] = fs_range_list
        fs_dict['m_list'] = m_list
        
        return fs_dict
    
    
    # wingband_from_fs()
    # Compute the frequency band we can measure with the given undersampling framerate.
    # fs:          Sampling frequency, i.e. camera framerate.
    # fw_center:    Desired center of the wingbeat frequencies. 
    # fw_min:        Lower frequency of band that can be measured containing fw_center.
    # fw_max:        Upper frequency of band.
    # bReversed:   If True, then the aliased frequencies will be in reverse order.
    #
    def wingband_from_fs(self, fs_lo, fs_hi, fw_center):
        # TODO: Note that this function does not work right.  It should return the inverse of self.fs_dict_from_wingband().
        bw_lo = fs_lo / 2.0
        
        fs = (fs_lo+fs_hi)/2.0
        
        if (fs != 0.0):
            # Find which multiples of fs/2 to use.
            n = np.round(fw_center / (fs/2))
            if (n*fs/2 < fw_center):
                fw_min = n*fs/2
                fw_max = (n+1)*fs/2
            else:
                fw_min = (n-1)*fs/2
                fw_max = n*fs/2
    
            bReversed = ((n%2)==1)
        else:
            fw_min = 0.0
            fw_max = 1.0
            bReversed = False
            
        return (fw_min, fw_max, bReversed)
        
        
    # get_baseband_range()
    # Get the baseband wingbeat alias frequency range for the given sampling frequency and m.
    def get_baseband_range(self, fs, m):
        
        kMax = int(np.floor(2*self.fw_center / m))
        
        # If m is even, then step down from fw in multiples of fs, keeping the last valid range above zero.
        if (m%2==0):
            for k in range(kMax):
                fbb_min_tmp = self.fw_min - k * fs   
                fbb_max_tmp = self.fw_max - k * fs
                if (fbb_min_tmp >= 0):
                    fbb_min = fbb_min_tmp   
                    fbb_max = fbb_max_tmp
                else:
                    break
                   
        else: # if m is odd, then step up from -fw in multiples of fs, keeping the first valid range above zero.
            for k in range(kMax):
                fbb_min = -self.fw_min + k * fs   
                fbb_max = -self.fw_max + k * fs
                if (fbb_max >= 0):
                    break
            
        return (fbb_min, fbb_max)
        
        
    # get_baseband_range_from_framerates()
    # Check if buffered framerates have stayed within an allowable 
    # range to make a valid measurement of the wingbeat passband.
    # If so, then compute the baseband frequency range
    #
    # Returns (bValid, [fbb_min, fbb_max])
    #
    def get_baseband_range_from_framerates(self, framerates):
        bValid = False

        fs_lo = np.min(framerates)
        fs_hi = np.max(framerates)
        #(fw_min, fw_max, bReversed) = self.wingband_from_fs(fs_lo, fs_hi, self.fw_center)
        for iRange in range(len(self.fs_dict['fs_range_list'])):
            (fs_min, fs_max) = self.fs_dict['fs_range_list'][iRange]
            if (fs_min < fs_lo < fs_hi < fs_max):
                bValid = True
                break

        m = self.fs_dict['m_list'][iRange]

        if (bValid):
            fs = np.mean(framerates)
            (fbb_min, fbb_max) = self.get_baseband_range(fs, m)
        else:
            fbb_min = 0.0
            fbb_max = np.Inf

        
        return (bValid, np.array([fbb_min, fbb_max]))
        
        
    # freq_from_intensity()
    # Get the wingbeat frequency by undersampling the image intensity, and then using 
    # an alias to get the image of the spectrum in a frequency band (typically 180-220hz).
    #
    # intensity:     The current pixel intensity.
    # fs:            The current framerate.
    #
    def freq_from_intensity(self, intensity, fs=0):            
        # Update the sample buffer.
        self.buffer[self.i]        = [intensity, fs]
        self.buffer[self.i+self.n] = [intensity, fs]

        # The buffered framerates.        
        framerates = self.buffer[(self.i+1):(self.i+1+self.n),1]
        
        # The baseband alias range.
        (bValid, fbb_range) = self.get_baseband_range_from_framerates(framerates)
        if (fbb_range[0] < fbb_range[1]):
            fbb_min = fbb_range[0]
            fbb_max = fbb_range[1]
            bReverse = False
        else:
            fbb_min = fbb_range[1]
            fbb_max = fbb_range[0]
            bReverse = True
                    
        # Get the wingbeat frequency.
        if (bValid):
            intensities = self.buffer[(self.i+1):(self.i+1+self.n),0]
            
#             # Multiplying by an alternating sequence has the effect of frequency-reversal in freq domain.
#             if (bReverse):
#                 a = np.ones(len(intensities))
#                 a[1:len(a):2] = -1
#                 intensities *= a

            # Get the dominant frequency, and shift it from baseband to wingband.
            fft = np.fft.rfft(intensities)
            fft[0] = 0                      # Ignore the DC component.
            i_max = np.argmax(np.abs(fft))  # Index of the dominant frequency.
            f_width = fbb_max - fbb_min     # Width of the passband.
            f_offset = np.abs(np.fft.fftfreq(self.n)[i_max]) * fs - fbb_min # * 2.0 * f_width    # Offset into the passband of the dominant freq.
            if (bReverse):
                freq = self.fw_max - f_offset
            else:
                freq = self.fw_min + f_offset

            #rospy.logwarn((fs, [fbb_min, fbb_max], bReverse, i_max, f_width, np.fft.fftfreq(self.n)[i_max], f_offset, freq))
        
        else:
            freq = 0.0
        
        # Go the the next sample slot.
        self.i += 1
        self.i %= self.n
        

        return freq
    
# End class WingbeatDetector            
    
    


