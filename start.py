import subprocess


if __name__ == "__main__": 
#        # creating threads 
    string1="cd Chat-bot & app.py"
    string2="npm start"
    sts = subprocess.Popen(string1, shell=True)
    sts2= subprocess.Popen(string2, shell=True)  