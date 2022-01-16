# confusion matrix in sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# actual values
actual = [1,0,0,1,0,0,1,0,0,1]
# predicted values
predicted = [1,0,0,1,0,0,0,1,0,0]

# confusion matrix
matrix = confusion_matrix(actual,predicted, labels=[1,0])

print(f"\n   Actual: {actual}")
print(f"Predicted: {predicted}")
# outcome values order in sklearn
tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
print(f"Outcome values : \ntrue positive={tp}, true negative={tn}, false positive={fp}, false negative={fn}\n")

# classification report for precision, recall f1-score and accuracy
matrix = classification_report(actual,predicted,labels=[1,0])
print(f"Classification report : \n{matrix}")
print('Macro average is the average of precision/recall/f1-score.\n')
print('Weighted average is just the weighted average of precision/recall/f1-score.\n')
