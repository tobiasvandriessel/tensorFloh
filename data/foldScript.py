import os
from random import randint

os.chdir("./UCF-101/")
if not os.path.exists("./folds"):
    os.makedirs("./folds")
    os.makedirs("./folds/1")
    os.makedirs("./folds/2")
    os.makedirs("./folds/3")
    os.makedirs("./folds/4")
    os.makedirs("./folds/5")


li = os.listdir()

print("dirs: ")
print(li)


setBrushes = [[], [], [], [], []]
setCuts = [[], [], [], [], []]
setJumps = [[], [], [], [], []]
setLunges = [[], [], [], [], []]
setPushes = [[], [], [], [], []]

for folder in li:
    if not folder == "folds":
        os.chdir(folder)
        print(folder)
        #print(os.listdir())
        for fil in os.listdir():
            if folder == "BrushingTeeth":
                setBrushes[randint(0,4)].append(fil)
            elif folder == "CuttingInKitchen":
                setCuts[randint(0,4)].append(fil)
            elif folder == "JumpingJack":
                setJumps[randint(0,4)].append(fil)
            elif folder == "Lunges":
                setLunges[randint(0,4)].append(fil)
            elif folder == "WallPushups":
                setPushes[randint(0,4)].append(fil)

        os.chdir("../")

#print(setBrushes[0])

def getSmallestSet(sets):
    elems = 10000
    indx = 10
    for idx, s in enumerate(sets):
        if len(s) < elems:
            elems = len(s)
            indx = idx

    return indx

for s in setBrushes:
    while len(s) > ((131/5) + 0.8):
        print("length was: " + str(len(s)))
        elem = s.pop()
        setBrushes[getSmallestSet(setBrushes)].append(elem)
        print("length is now: " + str(len(s)))
for s in setCuts:
    while len(s) > ((110/5) + 0.8):
        print("length was: " + str(len(s)))
        elem = s.pop()
        setCuts[getSmallestSet(setCuts)].append(elem)
        print("length is now: " + str(len(s)))
for s in setJumps:
    while len(s) > ((123/5) + 0.8):
        print("length was: " + str(len(s)))
        elem = s.pop()
        setJumps[getSmallestSet(setJumps)].append(elem)
        print("length is now: " + str(len(s)))
for s in setLunges:
    while len(s) > ((127/5) + 0.8):
        print("length was: " + str(len(s)))
        elem = s.pop()
        setLunges[getSmallestSet(setLunges)].append(elem)
        print("length is now: " + str(len(s)))
for s in setPushes:
    while len(s) > ((130/5) + 0.8):
        print("length was: " + str(len(s)))
        elem = s.pop()
        setPushes[getSmallestSet(setPushes)].append(elem)
        print("length is now: " + str(len(s)))


print("set Brushes lengths: ")
for s in setBrushes:
    print(len(s))
print("set Cuts lengths: ")
for s in setCuts:
    print(len(s))
print("set Cuts lengths: ")
for s in setJumps:
    print(len(s))
print("set Jumps lengths: ")
for s in setLunges:
    print(len(s))
print("set Lunges lengths: ")
for s in setPushes:
    print(len(s))


#"BrushingTeeth":
#"CuttingInKitchen":
#"JumpingJack":
#"Lunges":
#"WallPushups":


writecmds = []

for idx, s in enumerate(setBrushes):
    for name in s:
        #print(name)
        #break
        writecmds.append("move BrushingTeeth\\" + name + " folds\\" + str(idx+1) + "\\" + name + "\n")  

for idx, s in enumerate(setCuts):
    for name in s:
        #print(name)
        #break
        writecmds.append("move CuttingInKitchen\\" + name + " folds\\" + str(idx+1) + "\\" + name + "\n")  

for idx, s in enumerate(setJumps):
    for name in s:
        #print(name)
        #break
        writecmds.append("move JumpingJack\\" + name + " folds\\" + str(idx+1) + "\\" + name + "\n")  

for idx, s in enumerate(setLunges):
    for name in s:
        #print(name)
        #break
        writecmds.append("move Lunges\\" + name + " folds\\" + str(idx+1) + "\\" + name + "\n")  

for idx, s in enumerate(setPushes):
    for name in s:
        #print(name)
        #break
        writecmds.append("move WallPushups\\" + name + " folds\\" + str(idx+1) + "\\" + name + "\n")  


f = open('movecmds.txt', 'w')

for cmd in writecmds:
    f.write(cmd)
f.close()

print(os.getcwd())



#f = []
#for(dirpath, dirnames, filenames) in os.walk("./UCF-101"):
#    if dirnames == [] and os.path.dirname(dirpath) != "folds":
#        print("dir empty: " + dirpath)
#        print("dirname: " + os.path.dirname(dirpath + "\\"))
#        print("current work dir: " + os.getcwd())
#    f.extend(filenames)

#print(f)
