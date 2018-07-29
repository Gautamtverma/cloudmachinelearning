import os, csv

def writing_data_tocsv(imgData, gtData, writer):
    # print imgData
    count = 0
    for img in imgData:
        # print img
        pos = 0
        neg = 0
        for gt in gtData:
            #getting predicted image name and checking with ground truth image name
            if img[0] == gt[0]:
                print img
                # Removing all spaces, squares, single quotes from data
                words = img[1].strip('[').strip(']').strip(" ").split(',')
                w = []
                for word in words:
                    #after removing junk, adding all predicted labels of one image to w
                    w.append(word.strip("'").strip("'").strip(" ").strip("'"))
                # Removing all spaces, squares, single quotes from data
                GTwords = gt[1].strip('[').strip(']').strip(" ").split(',')
                gtw = []
                for word in GTwords:
                    # after removing junk, adding all original labels of one image to gtw
                    gtw.append(word.strip("'").strip("'").strip(" ").strip("'"))
                totpred = []
                for predword in w:
                    predword = predword.lower()
                    for gtword in gtw:
                        #comparing predicted label with original label
                        if predword == gtword:
                            totpred.append(predword)
                            pos += 1
                            count += 1
                        else:
                            neg += 1

                orgLabLen = len(gtw)
                predLabLen = len(totpred)

                if predLabLen == 0:
                    accuracy = 0
                    writer.writerow({'FileName': img[0], 'Predicted_Lable': w, 'Original_Label': gtw,
                                     'Predicted_Correct_Label': totpred, 'Accuracy': accuracy})

                else:
                    accuracy = float(predLabLen * 100.0 / orgLabLen)
                    writer.writerow({'FileName': img[0], 'Predicted_Lable': w, 'Original_Label': gtw,
                                     'Predicted_Correct_Label': totpred, 'Accuracy': accuracy})