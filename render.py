#!/usr/bin/env python
"""render.py

Description:
  Uses FFMPEG to create a video from bitmap files in the OUTPUT directory

Usage:
    render.py [options] --input=<val> --scale=<val> --output=<val>

    render.py -h | --help

Options:
  --input=<val>                 Prefix for all bitmap files
  --output=<val>                Prefix for output video
  --scale=<val>                 Resolution of resulting video: eg 500x500
  --sim                         Print resulting command to screen
"""

import os
from docopt import docopt

def main():
    # parse the command line options
    args = docopt(__doc__)

    # gather (required) args
    input_string = args['--input']
    scale_string = args['--scale']
    output_string = args['--output']
    
    call = 'ffmpeg -framerate 25 -i ./OUTPUT/%s_' % input_string
    call += '\%06d.bmp -c:v libx264 -vf fps=25,scale='
    call += '%s -pix_fmt yuv420p %s.mp4' % (scale_string, output_string)

    if args['--sim']:
        print(call)
    else:
        os.system(call)
# ----------------------------------------------------------------------
if __name__ == "__main__": 
    main()
