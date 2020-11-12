from Pro import StudentPerformance
import matplotlib.pyplot as plt
import itertools
import numpy as np
from PIL import ImageTk, Image
import threading
import time
import seaborn as sns
from tkinter import *
import pandas as pd
from tkinter.ttk import Progressbar
from tkinter.filedialog import askopenfilename
obj = StudentPerformance()
window = Tk()   
window.title("Student Performance Predictor")
window.state('normal')
window.configure(bg="White")
frame1 = Frame(window,bg="White")
frame2 = Frame(window,bg="White")
frame1.pack()
classlabels=['0','1']
def plot_confusion_matrix(cm,title, classes=classlabels,
                          cmap=plt.cm.Blues):

    plt.figure(figsize=(5,4.8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

#     print(cm)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.savefig(title+'.jpg')

def getCSV():
    csvfilename = askopenfilename(title = "Choose your CSV file")
    if(csvfilename != ""):
        Label(frame1, text = 'The file has been uploaded successfully!',bg="White",font =('Verdana', 15), fg="green").pack( pady=60 )        
        Button(frame1, text="Train Dataset",command = startTrainThread,width=20).pack(pady=40)
#         print(csvfilename)
        obj.readCsv(csvfilename)


    
def startTrainThread():
    progress.pack(pady=10)
    th=threading.Thread(target=train)
    th.start()

def train():
    obj.trainLR()
    progress['value'] = 33
    window.update_idletasks()
    time.sleep(2)
    obj.trainSVM()
    progress['value'] = 66
    window.update_idletasks()
    time.sleep(2)
    obj.trainRF()
    progress['value'] = 100
    window.update_idletasks()
    time.sleep(1)
    frame1.destroy()
    frame2.pack()
    
def test():
    predictFrame1.pack_forget()
    predictFrame2.pack_forget()
    predictFrame3.pack_forget()
    predictFrame4.pack_forget()
    predictFrame5.pack_forget()
    predictResultFrame.pack_forget()
    resultFrame.pack_forget()
    testFrame.pack()
    testPanel.pack()

def testClassifier():
    selector = var.get()
    if selector=='Random Forest':
        cm,a=obj.RandomForest()
    elif selector=='Logistic Regression':
        cm,a=obj.LogisticRegression()
    elif selector=='Support Vector Machine':
        cm,a=obj.SVM()
    acc_label.config(text="Accuracy :" +str(a))
    plot_confusion_matrix(cm,title=selector)
    from PIL import ImageTk, Image
    im = Image.open(selector+'.png')
    newsize=(500,500)
    im = im.resize(newsize)
    img = ImageTk.PhotoImage(im)
    cm_label.config(image=img)
    cm_label.image=img
    resultFrame.pack()

#     print(cm,a,precision,recall,f1)
        
        

        
def predict():
    testFrame.pack_forget()
    testPanel.pack_forget()
    resultFrame.pack_forget()
    predictResultFrame.pack_forget()
    predictFrame1.pack()
    predictFrame2.pack()
    predictFrame3.pack()
    predictFrame4.pack()
    predictFrame5.pack()
    
def predictResult():
    result = obj.predict(gender.get(),race.get(),edu.get(),prep.get(),int(reading.get()),int(writing.get()))
     
    if(result[1][0]==0):
        lrY = "Passed"
    else:
        lrY = "failed"
    if(result[0][0]==0):
        rfY = "Passed"
    else:
        rfY = "failed"
    if(result[2][0]==0):
        svmY = "Passed"
    else:
        svmY = "failed"
    
    logistic_regression_label.config(text="Logistic Regression : "+lrY)
    random_forest_label.config(text="Random Forest : "+rfY)
    SVM_label.config(text="Support Vector Machine : "+svmY)
    
    predictResultFrame.pack()




#Frame 1
Label(frame1, text = 'Choose the dataset to work upon',bg="White",font =('Verdana', 15)).pack( pady=40 )
Button(frame1, text="Choose your CSV file", command=getCSV,width=20).pack(pady=40)
progress = Progressbar(frame1, orient = HORIZONTAL, length = 500, mode = 'determinate')


#Frame 2
Button(frame2, text="Test",command=test,width=20 ).pack(pady=40,side=LEFT,padx=175)
Button(frame2, text="Predict",command=predict,width=20).pack(pady=40,side=RIGHT,padx=175)


#TestFrame
testFrame=Frame(window,bg="White")
var = StringVar()
var.set('Random Forest')
choices = { 'Random Forest','Logistic Regression','Support Vector Machine'}
popupMenu = OptionMenu(testFrame, var, *choices)
Label(testFrame, text = 'Testing the data',bg="White",font =('Verdana', 15)).pack(pady=20)
Label(testFrame, text = 'Select Classifier',bg="White",font =('Verdana', 15)).pack( side=LEFT,padx=20,pady=10 )
popupMenu.pack(side=LEFT)

#TestPanel
testPanel=Frame(window,bg="White")
Button(testPanel, text="Test",command=testClassifier,width=20).pack(pady=20)

#TestResultFrame
resultFrame=Frame(window,bg='white',pady=10)
acc_label = Label(resultFrame, text = "",bg="White",font =('Verdana', 10))
acc_label.pack( pady=10 )
from PIL import ImageTk, Image
# img = ImageTk.PhotoImage(Image.open("mkbhd1.jpg"))
cm_label = Label(resultFrame)
# cm_label.image=img
cm_label.pack(pady=10)

#predictFrame
predictFrame1=Frame(window,bg="White",pady=10)
predictFrame2=Frame(window,bg="White",pady=10)
predictFrame3=Frame(window,bg="White",pady=10)
predictFrame4=Frame(window,bg="White",pady=10)
predictFrame5=Frame(window,bg="White",pady=10)

gender = StringVar()
race = StringVar()
edu = StringVar()
prep = StringVar()
reading = StringVar()
writing = StringVar()

gender.set('male')
choices1 = { 'male', 'female'}
popupMenu1 = OptionMenu(predictFrame1, gender, *choices1)
Label(predictFrame1, text = 'Select Gender',bg="White",font =('Verdana', 12)).pack( side=LEFT,padx=20,pady=10 )
popupMenu1.pack(side=LEFT)
race.set('group A')
choices2 = {'group A', 'group B', 'group C', 'group D', 'group E'}
popupMenu2 = OptionMenu(predictFrame1, race, *choices2)
Label(predictFrame1, text = 'Select Race/Ethnicity',bg="White",font =('Verdana', 12)).pack( side=LEFT,padx=20,pady=10 )
popupMenu2.pack(side=LEFT)
edu.set('high school')
choices3 = {"associate's degree","bachelor's degree",'high school',"master's degree",'some college','some high school'}
popupMenu3 = OptionMenu(predictFrame2, edu, *choices3)
Label(predictFrame2, text = 'Select Parental education level',bg="White",font =('Verdana', 12)).pack( side=LEFT,padx=20,pady=10 )
popupMenu3.pack(side=LEFT)
prep.set('none')
choices4 = {'none', 'completed'}
popupMenu4 = OptionMenu(predictFrame2, prep, *choices4)
Label(predictFrame2, text = 'Course Preparation',bg="White",font =('Verdana', 12)).pack( side=LEFT,padx=20,pady=10 )
popupMenu4.pack(side=LEFT)
Label(predictFrame3, text = 'Reading_Score',bg="White").pack( pady=10,side=LEFT,padx=10)
Reading_Score=Entry(predictFrame3,textvariable=reading).pack(pady=10,side=LEFT,padx=15)
Label(predictFrame3, text = 'Writing_Score',bg="White").pack( pady=10,side=LEFT,padx=10)
Writing_Score=Entry(predictFrame3,textvariable=writing).pack(pady=10,side=LEFT,padx=15)
Button(predictFrame5, text="Predict",command=predictResult,width=20).pack(pady=20)

#PredictResultFrame
predictResultFrame=Frame(window,bg='white',pady=10)
logistic_regression_label = Label(predictResultFrame, text = "",bg="White",font =('Verdana', 10))
logistic_regression_label.pack( pady=10 )
random_forest_label = Label(predictResultFrame, text = "",bg="White",font =('Verdana', 10))
random_forest_label.pack( pady=10 )
SVM_label = Label(predictResultFrame, text = "",bg="White",font =('Verdana', 10))
SVM_label.pack( pady=10 )
window.mainloop()  