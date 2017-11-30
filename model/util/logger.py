import datetime
import sys


class Logger(object):
    def __init__(self, log_path=None):
        self.log_path = log_path

    def Log(self, message):
        if self.log_path:
            # Write to the log file then close it
            with open(self.log_path, 'a') as f:
                datetime_string = datetime.datetime.now().strftime(
                    "%y-%m-%d %H:%M:%S")
                f.write("%s %s\n" % (datetime_string, message))