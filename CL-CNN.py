# zhangtianxia nankai
#2020-3-20
import scikitplot as skplt
import torch
import torch.utils.data as Data
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F 
# parameters setting
learning_rate = 0.05
batch_size = 64
epoch = 100

#load dataset
x_train = np.load("clientx.npy").reshape(-1,1,20,25)
y_train = np.load("clienty.npy").ravel()
x_train = torch.from_numpy(x_train)
x_train = x_train.type(torch.FloatTensor)
y_train = torch.from_numpy(y_train)


vail_x = np.load('valid_x.npy').reshape(-1,1,20,25)
vail_y = np.load('valid_y.npy').ravel()
vail_label = vail_y
vail_x = torch.from_numpy(vail_x)
vail_x = vail_x.type(torch.FloatTensor)
vail_y = torch.from_numpy(vail_y)

x_test = np.load('test_x.npy').reshape(-1,1,20,25)
y_test = np.load('test_y.npy').ravel()
test_label = y_test
x_test = torch.from_numpy(x_test)
x_test = x_test.type(torch.FloatTensor)
y_test = torch.from_numpy(y_test)


def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x
def test(model, loader):
    model.eval()
    total = 0
    accuracy = 0
    test_loss = 0
    with torch.no_grad():
        for images,labels in loader:
            images = get_variable(images)
            labels = get_variable(labels) 
            outputs = model(images)
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
            test_loss += criterion(outputs, labels.long())
        
    test_loss = test_loss/(len(loader))
    accuracy = accuracy/total    

    return test_loss,accuracy

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),  
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear(960,128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128,10),)
          
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)  # reshape 
        out = self.fc(out)

        return out


# data for pytorch format
train_dataset = Data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True)

vail_data = Data.TensorDataset(vail_x, vail_y)
vailloader = torch.utils.data.DataLoader(vail_data, shuffle=False, batch_size=len(vail_x))

test_dataset = Data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=len(test_dataset),shuffle=False)


cnn = CNN() # generate a cnn model 
if torch.cuda.is_available():
    cnn = cnn.cuda() # load the model to GPU 
    
criterion = nn.CrossEntropyLoss() # select loss function 
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.5) # select optimizer


train_losses, test_losses,accuracy_rates,atest_losses,ae = [],[],[],[],[]; # monitor list
# train model
for e in range(epoch):
    train_loss = 0
    test_loss = 0
    accuracy = 0
    train_loss_1 = 0
    for i, (images, labels) in enumerate(train_loader):
        images = get_variable(images)
        labels = get_variable(labels)  
        outputs = cnn(images)
        optimizer.zero_grad()  
        loss = criterion(outputs, labels.long())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()


    if e <= 10:
        if (e) % 2 == 0:
            train_loss,_ = test(cnn, vailloader)
            test_loss,accuracy = test(cnn, train_loader)
            print("Epoch: {}/{}.. ".format(e+1,epoch),
          "Training Loss: {:.5f}.. ".format(train_loss),
          "Test Loss: {:.5f}.. ".format(test_loss),
          "Test Accuracy: {:.5f}".format(accuracy))
            ae.append(e+1)
            accuracy_rates.append(accuracy)
            atest_losses.append(test_loss)
    else:
        if (e+1)% 10 == 0:
            train_loss,_ = test(cnn, train_loader)
            test_loss,accuracy = test(cnn, vailloader)
            print("Epoch: {}/{}.. ".format(e+1,epoch),
          "Training Loss: {:.5f}.. ".format(train_loss),
          "Test Loss: {:.5f}.. ".format(test_loss),
          "Test Accuracy: {:.5f}".format(accuracy))
            ae.append(e+1)
            accuracy_rates.append(accuracy)
            atest_losses.append(test_loss)     

# save model
torch.save(cnn.state_dict(),'CNN.pth')

# evalute model
descriptions = (
                "B1",
                "B2",
                "B3",
                "IR1",
                "IR2",
                "IR3",
                "OR1",
                "OR2",
                "OR3",
                "N0",
                )

label_decoder = dict(zip(range(10), descriptions))
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
print("|Item|Accuracy (%)|")
print("|-+-|")
with torch.no_grad():
    cnn.eval()
    for images, labels in test_loader:
        images = get_variable(images)
        labels = get_variable(labels) 
        outputs = cnn(images)
        prob = torch.exp(outputs)
        top_probs, top_classes = prob.topk(1, dim=1)
        c = (labels.long() == top_classes.view(labels.shape)).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label.item()] += c[i].item()
            class_total[label.item()] += 1


for i in range(10):
    print('|{}|{:.1f}'.format(
        label_decoder[i], 100 * class_correct[i] / class_total[i]))
plt.legend()

from sklearn.metrics import classification_report
outputs = F.softmax(outputs,dim=1)
y = outputs.cpu().numpy()
inverted = np.argmax(y,axis=1)
print (classification_report(test_label, inverted, digits=4))


from sklearn.metrics import confusion_matrix
C=confusion_matrix(test_label, inverted)
print(C, end='\n\n')

skplt.metrics.plot_confusion_matrix(y_true=test_label, y_pred=inverted)
skplt.metrics.plot_precision_recall(test_label, y)

