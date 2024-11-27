# ESP32 Implementation Details Discussion

- Data coming in from the analog sensor is 16-bit PCM, so it may be easier to work entirely in integer math? (Kill me)

## Peak-finding Algorithm

1. Find all points whereby A < B > C. Record their location and height
2. Iterate through lowest to highest, filtering based on criteria
   - Min height: must be at least (3x stdev?)
   - Min distance: must be (0.5 - 3.0 Hz) from other peaks
     - A better implementation would also have min prominence, and leave distance filtering for later. This allows 
       eg. 0.1-10 Hz to be detectable, and then the software can handle that later
3. Return filtered results (find average distance between peaks, and then just calculate BPM very simply)

## FFT

- If the FFT being done is a power of 4 (subset of being a power of 2), then the performance is increased further
  - `dsps_fft4r_(fc32/sc16)_ae32` functions


