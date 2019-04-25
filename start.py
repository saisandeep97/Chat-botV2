import subprocess
import time


if __name__ == "__main__": 
#        # creating threads 
    string1="cd Chat-bot & python app.py"
    string2="npm start"
    sts = subprocess.Popen(string1, shell=True)
    sts2= subprocess.Popen(string2, shell=True) 
    time.sleep(5)
    Call_URL = "http://localhost:3000"
    mycmd = r'start chrome /new-tab {}'.format(Call_URL)
    sts3=subprocess.Popen(mycmd,shell = True)  