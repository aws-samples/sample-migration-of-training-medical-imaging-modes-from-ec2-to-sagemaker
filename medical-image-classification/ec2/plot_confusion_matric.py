from sklearn.metrics import confusion_matrix
import json
import numpy as np
import matplotlib.pyplot as plt

# load the confusion matrix
confusion_matrix = np.load('confusion_matrix.npy')

# load the labels file 
labels_file = './label_dict.json'
label_name_list = []
with open(labels_file, 'r', encoding='utf-8') as fid:
    for key in json.load(fid).keys():
        label_name_list.append(key)

# save the confusion matrix as a png file
# write the percentages in the confusion matrix
for i in range(8):
    for j in range(8):
        plt.text(j, i, '{:.2f}'.format(confusion_matrix[i, j]), ha='center', va='center')
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(label_name_list))
plt.xticks(tick_marks, label_name_list, rotation=90)
plt.yticks(tick_marks, label_name_list)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix.png')