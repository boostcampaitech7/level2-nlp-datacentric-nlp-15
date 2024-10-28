# use subprocess to run the code
import subprocess
import os
# set chdir to src
os.chdir("src")

subprocess.run(["python", "main.py"])