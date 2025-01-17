
import cv2
import numpy as np
from lib import load_model, detect_lp, im2single

import base64

# Ham sap xep contour tu trai sang phai
def sort_contours(cnts):

    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

# Dinh nghia cac ky tu tren bien so
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ-'



# Đường dẫn ảnh
#img_path = "test/8966.jpg"

# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString

# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
Dmax = 608
Dmin = 288
# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

# Cau hinh tham so cho model SVM
digit_w = 30 # Kich thuoc ki tu
digit_h = 60 # Kich thuoc ki tu

model_svm = cv2.ml.SVM_load('svm.xml')
def detection_SVM(Ivehicle):
    # Lấy kích thước ảnh đầu vào
    Height, Width = Ivehicle.shape[:2]

    # Tính tỷ lệ giữa chiều rộng và chiều cao của ảnh và tìm kích thước nhỏ nhất
    ratio = float(max(Height, Width)) / min(Height, Width)
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    # Sử dụng mô hình để phát hiện biển số
    _, LpImg, _ = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)
   
    if len(LpImg) > 0:
        # Chuyển đổi ảnh biển số sang grayscale
        LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
        roi = LpImg[0]
        roi = cv2.resize(roi, (600, 288))
        # Chuyển ảnh sang ảnh grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Áp dụng ngưỡng để phân tách số và nền
        binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
        
        # Tiến hành phân đoạn kí tự
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        contours, _ = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        plate_info = ""

        # Hiển thị khung các contour tìm thấy
        for c in sort_contours(contours):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h / w
            if 1.5 <= ratio <= 3.5 and h / roi.shape[0] >= 0.6:
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                

                # Tách số và dự đoán
                curr_num = thre_mor[y:y + h, x:x + w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                curr_num = np.array(curr_num, dtype=np.float32)
                curr_num = curr_num.reshape(-1, digit_w * digit_h)

                # Dự đoán số hoặc ký tự
                result = model_svm.predict(curr_num)[1]
                result = int(result[0, 0])

                if result <= 9:
                    result = str(result)
                else:
                    result = chr(result)

                plate_info += result

    return plate_info

#cv2.imshow("Cac contour tim duoc", roi)
#cv2.waitKey()

#     # Viet bien so len anh
#     cv2.putText(Ivehicle,fine_tune(plate_info),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)

#     # Hien thi anh
#     print("Bien so=", plate_info)
#     cv2.imshow("Hinh anh output",Ivehicle)
#     cv2.waitKey()



# cv2.destroyAllWindows()

def detection1line_SVM(plate_info , roi):
    if (len(roi)):
        roi = cv2.resize(roi, (220, 100))
        
        # Chuyen anh bien so ve gray
        gray = cv2.cvtColor( roi, cv2.COLOR_BGR2GRAY)

        # Ap dung threshold de phan tach so va nen
        binary = cv2.threshold(gray, 127, 255,
                             cv2.THRESH_BINARY_INV)[1]

        #cv2.imshow("1",binary)
        #cv2.waitKey()


        # Segment kí tự
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        cont, _  = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


        for c in sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h/w
            if 1.5<=ratio<=3.5: # Chon cac contour dam bao ve ratio w/h
                if h/roi.shape[0]>=0.6: # Chon cac contour cao tu 60% bien so tro len

                    # Ve khung chu nhat quanh so
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    
                    # Tach so va predict
                    curr_num = thre_mor[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                    curr_num = np.array(curr_num,dtype=np.float32)
                    curr_num = curr_num.reshape(-1, digit_w * digit_h)

                    # Dua vao model SVM
                    result = model_svm.predict(curr_num)[1]
                    result = int(result[0, 0])

                    if result<=9: # Neu la so thi hien thi luon
                        result = str(result)
                    else: #Neu la chu thi chuyen bang ASCII
                        result = chr(result)

                    plate_info +=result
        #cv2.imshow("2", roi)
        #cv2.waitKey()

    return plate_info


def detection2line_SVM(Ivehicle):

    # Đường dẫn ảnh,
    #img_path = "test/0202_04179_b.jpg"

    # Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
    Dmax = 608
    Dmin = 288

    # Chia cat anh lam doi de nhan dien anh 2 dong

    # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    _, LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)



    # Chuyen doi anh bien so
    LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
    roiTemp = LpImg[0]

    height = int(roiTemp.shape[0] / 2)
    roi1 = roiTemp[0:height, :]
    roi2 = roiTemp[height:roiTemp.shape[0], :]
    

    plate_info = ""
    plate_info=detection1line_SVM(plate_info, roi1)


    plate_info = plate_info + "-"
    plate_info= detection1line_SVM(plate_info, roi2)

    return plate_info

    

image_path = "test/41.jpg"

image = cv2.imread(image_path)  # Đọc ảnh từ đường dẫn
result = detection2line_SVM(image)
print("Biển số:", result)