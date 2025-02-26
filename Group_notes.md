The suggested method to view the camera stream from the manual does not work. There is a temporary solution until the TA's announce an official solution:

Try this command from paparazzi folder (or modify path to sdp file):

```ffplay -i ./sw/tools/rtp_viewer/rtp_5000.sdp -protocol_whitelist "file,crypto,data,rtp,udp" -fflags nobuffer -flags low_delay```