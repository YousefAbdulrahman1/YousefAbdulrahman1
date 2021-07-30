from numba import njit, prange
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import multiprocessing




def recognition():
    cap = cv.VideoCapture("VID-20210122-WA0003.mp4")
    whT = 320
    confThreshold = 0.9
    nmsThreshold = 0.2
    #Y = []

    xs = 0

    #### LOAD MODEL
    ## Coco Names
    classesFile = "obj.names"
    classNames = []
    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    #print(classNames)
    ## Model Files

    modelConfiguration = "yolov3_custom.cfg"
    modelWeights = "yolov3_custom_final_2.weights"
    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


    def findObjects(outputs, img):
        hT, wT, cT = img.shape
        bbox = []
        classIds = []
        confs = []
        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    w, h = int(det[2] * wT), int(det[3] * hT)
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    classIds.append(classId)
                    confs.append(float(confidence))

        indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
        #print(f"ind = {len(indices)}")
        #tobesorted = []
        numbers = []
        Xs = []
        Analog = []
        Analog2 = []
        sortednumbers = []
        #counter = 0
        #print(f"indices = {indices}")
        for i in indices:

            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]

            #print(x,y,w,h)




            if len(indices) > 1:

                if True:
                    # If passed here then it's verified frame

                    Xs.append(x)
                    numbers.append(int(classNames[classIds[i]]))


                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                               (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    #print(int(classNames[classIds[i]]))






                else:
                    pass


        J = 0



        for number in numbers:
            Analog.append((Xs[J], numbers[J]))
            J += 1
        Analog.sort()
        Analog.reverse()

        counter = 0
        sum = 0
        sumlist = []
        for anumber in Analog:
            if counter == 0:
                sum =Analog[counter][1]
                sumlist.append(sum)
            elif counter == 1:
                sum =Analog[counter][1]*10
                sumlist.append(sum)
            elif counter == 2:
                sum = Analog[counter][1]*100
                sumlist.append(sum)
            elif counter == 3:
                sum = Analog[counter][1]*1000
                sumlist.append(sum)
            elif counter == 4:
                sum = Analog[counter][1]*10000
                sumlist.append(sum)
            counter += 1

        finalres = 0
        counter2 = 0
        for s in sumlist:
            finalres += sumlist[counter2]
            counter2 += 1


        print(finalres)

        if xs == 0:
            with open("plot_data.txt", 'a') as txtfile:
                txtfile.write(str(0) + "," + str(0) + '\n')
        with open("plot_data.txt", 'a') as txtfile:
            txtfile.write(str(finalres) + "," + str(xs) + '\n')



        print(finalres)

        if xs == 0:
            with open("plot_data.txt", 'a') as txtfile:
                txtfile.write(str(0) + "," + str(0) + '\n')
        with open("plot_data.txt", 'a') as txtfile:
            txtfile.write(str(finalres) + "," + str(xs) + '\n')






    while True:
        timer = cv.getTickCount()

        success, img = cap.read()

        blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        findObjects(outputs, img)

        cv.imshow('Image', img)
        xs += 1

        fps = cv.getTickFrequency()/(cv.getTickCount()-timer)
        print(fps)
        cv.waitKey(1)







def plotting():
    fig = plt.figure()
    axl = fig.add_subplot(1, 1, 1)
    def animate(k):
        graph_data = open('plot_data.txt', 'r').read()
        lines = graph_data.split('\n')
        Xs = []
        Ys = []
        for line in lines:
            if len(line) > 1:
                x, y = line.split(",")
                Xs.append(int(x))
                Ys.append(int(y))
        axl.clear()
        axl.plot(Ys, Xs)



    ani = animation.FuncAnimation(fig ,animate)
    plt.show()

p1 = multiprocessing.Process(target = recognition)
p2 = multiprocessing.Process(target = plotting)



if __name__ == "__main__":
    p1.start()
    p2.start()
