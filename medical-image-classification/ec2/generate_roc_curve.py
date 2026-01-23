# Generate ROC curve and AUC for a given model and dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from model_def import ModelDef
from train import create_data_loaders
import torch
from sklearn.metrics import confusion_matrix
import json
# softmax
from tqdm import tqdm
from torch.nn import Softmax


def main():
    model_name='DenseNet121'
    model_file = '/home/ubuntu/data/vindr-spinexr/spine-output/' + model_name +'/model.pth'
    data_dir = '/home/ubuntu/data/vindr-spinexr'
    # Load the model
    model = ModelDef(num_classes=8, model_name=model_name).get_model()
    model.load_state_dict(torch.load(model_file))
    
    train_loader, val_loader, test_loader, _ = create_data_loaders(data=data_dir, batch_size=20)
   
    # load the labels file 
    labels_file = './label_dict.json'
    with open(labels_file, 'r', encoding='utf-8') as fid:
        label_dict = json.load(fid)
    label_name_list = []
    
    for key in label_dict.keys():
        label_name_list.append(key)
    fid.close()
    
    print(label_name_list)
    
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    file_name_list = []
    with torch.no_grad():
        model.eval()
        all_labels = []
        all_outputs = []
        count = 0
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            outputs = model(inputs[:,:,:,:,0])
            # apply softmax 
            softmax_out = Softmax(dim=1)(outputs)
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(softmax_out.cpu().numpy())
            _file_name = test_loader.dataset.image_files[count].split('/')[-1]
            # print(outputs[0])
            count +=1
            file_name_list.append(_file_name)
            
    
    # calculate the accuracy for each class
    for i in range(8):
        count = 0
        for j in range(len(file_name_list)):
            if label_name_list[i] in file_name_list[j]:
                count += 1
        print(label_name_list[i], count)
    
    

    accuracy_per_class = np.zeros(8)
    
    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    all_labels = label_binarize(all_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7])
    # Generate ROC curve and AUC for a given model and dataset
    for i in range(8):
        fpr, tpr, _ = roc_curve(all_labels[:,i], all_outputs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, \
            label='%s (area = %0.2f)' % (label_name_list[i], roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Receiver Operating Characteristic', fontsize=8)
    plt.legend(loc='lower right', fontsize=8)
    plt.savefig('roc_curve.png')
    
    with open('accuracy_per_class.txt', 'a', encoding='utf-8') as write_file:
        for i in range(8):
            accuracy_per_class[i] = np.sum((all_outputs[:, i] > 0.5) == all_labels[:, i]) / len(all_labels)
            write_file.write('%s\t%0.2f\n' % (label_name_list[i], accuracy_per_class[i]))
    write_file.close()  

    
    # Calculate confusion matrix
    y_pred = np.argmax(all_outputs, axis=1)
    y_true = np.argmax(all_labels, axis=1)
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    # save confusion matrix as a numpy array
    np.save('confusion_matrix.npy', cm)

    

    
if __name__ == "__main__":
    main()
    