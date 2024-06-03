import cv2
import os
import numpy as np
import glob

digit_w = 30
digit_h = 60

write_path = "data/"

def get_digit_data(path):
    digit_list = []
    label_list = []

    for number in range(10):
        for img_org_path in glob.iglob(path + str(number) + '/*.jpg'):
            img = cv2.imread(img_org_path, 0)
            img = np.array(img)
            img = img.reshape(-1, digit_h * digit_w)
            digit_list.append(img)
            label_list.append([int(number)])

    for number in range(65, 91):
        for img_org_path in glob.iglob(path + str(number) + '/*.jpg'):
            img = cv2.imread(img_org_path, 0)
            img = np.array(img)
            img = img.reshape(-1, digit_h * digit_w)
            digit_list.append(img)
            label_list.append([int(number)])

    return np.array(digit_list, dtype=np.float32).reshape(-1, digit_h * digit_w), np.array(label_list).reshape(-1)


# Lấy dữ liệu
digit_path = "data/"
digit_array, label_list = get_digit_data(digit_path)

# Khởi tạo và huấn luyện mô hình SVM
svm_model = cv2.ml.SVM_create()
svm_model.setType(cv2.ml.SVM_C_SVC)
svm_model.setKernel(cv2.ml.SVM_INTER)
svm_model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
svm_model.train(digit_array, cv2.ml.ROW_SAMPLE, label_list)

# Lưu mô hình SVM
svm_model.save("svm.xml")
def calculate_precision_recall_f1score(digit_array, label_list):
    TP = FP = FN = 0

    for i in range(len(label_list)):
        predicted_label = predict_svm(digit_array[i])
        actual_label = label_list[i]

        if predicted_label == actual_label:
            if actual_label == 1:
                TP += 1
        else:
            if actual_label == 1:
                FN += 1
            else:
                FP += 1

    precision = (TP / (TP + FP)) * 100
    recall = (TP / (TP + FN)) * 100
    f1_score = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1_score
def predict_svm(sample):
    # Sử dụng mô hình SVM để dự đoán nhãn của mẫu dữ liệu
    result = svm_model.predict(sample.reshape(1, -1))[1]
    predicted_label = int(result[0, 0])
    return predicted_label
# Calculate precision, recall, and F1-score
precision, recall, f1_score = calculate_precision_recall_f1score(digit_array, label_list)

print("Precision (%):", precision)
print("Recall (%):", recall)
print("F1-score (%):", f1_score)
