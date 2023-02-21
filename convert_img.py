import os

list = "./img/car"

for count, filename in enumerate(os.listdir(list)):
    dst = f"car{str(count)}.jpg"
    src =f"{list}/{filename}"  # foldername/filename, if .py file is outside folder
    dst =f"{list}/{dst}"
        
    # rename() function will
    # rename all the files
    os.rename(src, dst)
