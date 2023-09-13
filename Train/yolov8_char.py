#***************************************************************
# Document Understanding using YOLO8
#***************************************************************
# https://docs.ultralytics.com/modes/predict/#working-with-results
# https://encord.com/blog/yolo-object-detection-guide/          # YOLO8 Explanation

# conda create -n yolov8_ocr python=3.10 tk
# conda activate yolov8_ocr
# conda install -c conda-forge -c pytorch -c nvidia ultralytics pytorch torchvision pytorch-cuda=11.8 
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# conda install -c conda-forge charset-normalizer
# conda install opencv

from ultralytics import YOLO
import os, sys, torch
import gpu_status
import cv2, tkinter

os.environ['KMP_DUPLICATE_LIB_OK']='True'   # Library Collision issue

cBlue=(255,0,0); cRed=(0,0,255); cOrange=(0,128,255); cDarkGray=(80,80,80); cMagenta=(255,0,255); cCyan=(255,255,0)
cGreen=(0,255,0); cYellow=(0,255,255); cLightGray=(160,160,160)
'''
clsType = ['Word', 'TextLine', 'TextBlock', 'Column', 'NumExpression', 'Symbol', 
           'Table', 'Picture', 'HorLine', 'VertLine',  'BoxLine', 'Column', 'Page' ]        
'''
clsColor = [cBlue, cRed, cOrange, cDarkGray, cMagenta, cCyan, 
            cGreen, cYellow, cLightGray, cLightGray,  cLightGray, cLightGray, cDarkGray]
# clsType = [        
# '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', 

# '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', 

# '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266', '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280', '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294', '295', '296', '297', '298', '299', 

# '300', '301', '302', '303', '304', '305', '306', '307', '308', '309', '310', '311', '312', '313', '314', '315', '316', '317', '318', '319', '320', '321', '322', '323', '324', '325', '326', '327', '328', '329', '330', '331', '332', '333', '334', '335', '336', '337', '338', '339', '340', '341', '342', '343', '344', '345', '346', '347', '348', '349', '350', '351', '352', '353', '354', '355', '356', '357', '358', '359', '360', '361', '362', '363', '364', '365', '366', '367', '368', '369', '370', '371', '372', '373', '374', '375', '376', '377', '378', '379', '380', '381', '382', '383', '384', '385', '386', '387', '388', '389', '390', '391', '392', '393', '394', '395', '396', '397', '398', '399', 

# '400', '401', '402', '403', '404', '405', '406', '407', '408', '409', '410', '411', '412', '413', '414', '415', '416', '417', '418', '419', '420', '421', '422', '423', '424', '425', '426', '427', '428', '429', '430', '431', '432', '433', '434', '435', '436', '437', '438', '439', '440', '441', '442', '443', '444', '445', '446', '447', '448', '449', '450', '451', '452', '453', '454', '455', '456', '457', '458', '459', '460', '461', '462', '463', '464', '465', '466', '467', '468', '469', '470', '471', '472', '473', '474', '475', '476', '477', '478', '479', '480', '481', '482', '483', '484', '485', '486', '487', '488', '489', '490', '491', '492', '493', '494', '495', '496', '497', '498', '499', 

# '500', '501', '502', '503', '504', '505', '506', '507', '508', '509', '510', '511', '512', '513', '514', '515', '516', '517', '518', '519', '520', '521', '522', '523', '524', '525', '526', '527', '528', '529', '530', '531', '532', '533', '534', '535', '536', '537', '538', '539', '540', '541', '542', '543', '544', '545', '546', '547', '548', '549', '550', '551', '552', '553', '554', '555', '556', '557', '558', '559', '560', '561', '562', '563', '564', '565', '566', '567', '568', '569', '570', '571', '572', '573', '574', '575', '576', '577', '578', '579', '580', '581', '582', '583', '584', '585', '586', '587', '588', '589', '590', '591', '592', '593', '594', '595', '596', '597', '598', '599', 

# '600', '601', '602', '603', '604', '605', '606', '607', '608', '609', '610', '611', '612', '613', '614', '615', '616', '617', '618', '619', '620', '621', '622', '623', '624', '625', '626', '627', '628', '629', '630', '631', '632', '633', '634', '635', '636', '637', '638', '639', '640', '641', '642', '643', '644', '645', '646', '647', '648', '649', '650', '651', '652', '653', '654', '655', '656', '657', '658', '659', '660', '661', '662', '663', '664', '665', '666', '667', '668', '669', '670', '671', '672', '673', '674', '675', '676', '677', '678', '679', '680', '681', '682', '683', '684', '685', '686', '687', '688', '689', '690', '691', '692', '693', '694', '695', '696', '697', '698', '699', 

# '700', '701', '702', '703', '704', '705', '706', '707', '708', '709', '710', '711', '712', '713', '714', '715', '716', '717', '718', '719', '720', '721', '722', '723', '724', '725', '726', '727', '728', '729', '730', '731', '732', '733', '734', '735', '736', '737', '738', '739', '740', '741', '742', '743', '744', '745', '746', '747', '748', '749', '750', '751', '752', '753', '754', '755', '756', '757', '758', '759', '760', '761', '762', '763', '764', '765', '766', '767', '768', '769', '770', '771', '772', '773', '774', '775', '776', '777', '778', '779', '780', '781', '782', '783', '784', '785', '786', '787', '788', '789', '790', '791', '792', '793', '794', '795', '796', '797', '798', '799', 

# '800', '801', '802', '803', '804', '805', '806', '807', '808', '809', '810', '811', '812', '813', '814', '815', '816', '817', '818', '819', '820', '821', '822', '823', '824', '825', '826', '827', '828', '829', '830', '831', '832', '833', '834', '835', '836', '837', '838', '839', '840', '841', '842', '843', '844', '845', '846', '847', '848', '849', '850', '851', '852', '853', '854', '855', '856', '857', '858', '859', '860', '861', '862', '863', '864', '865', '866', '867', '868', '869', '870', '871', '872', '873', '874', '875', '876', '877', '878', '879', '880', '881', '882', '883', '884', '885', '886', '887', '888', '889', '890', '891', '892', '893', '894', '895', '896', '897', '898', '899', 

# '900', '901', '902', '903', '904', '905', '906', '907', '908', '909', '910', '911', '912', '913', '914', '915', '916', '917', '918', '919', '920', '921', '922', '923', '924', '925', '926', '927', '928', '929', '930', '931', '932', '933', '934', '935', '936', '937', '938', '939', '940', '941', '942', '943', '944', '945', '946', '947', '948', '949', '950', '951', '952', '953', '954', '955', '956', '957', '958', '959', '960', '961', '962', '963', '964', '965', '966', '967', '968', '969', '970', '971', '972', '973', '974', '975', '976', '977', '978', '979', '980', '981', '982', '983', '984', '985', '986', '987', '988', '989', '990', '991', '992', '993', '994', '995', '996', '997', '998', '999'
# ]
clsType = [        
'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', 

'100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', 
]
#=========================================================
# Get Screen Info.
#=========================================================
def get_screen_resolution():
    global screenWidth, screenHeight

    app = tkinter.Tk()
    screenWidth = app.winfo_screenwidth()
    screenHeight = app.winfo_screenheight()

    screenWidth = int(round((screenWidth * 2.0) /3.0,0))
    #print("width=%d, height=%d" %(width,height))
    app.destroy()

def show_program_usage():
    print("*"*80)
    print("USAGE[train]:   python yolov8.py  train    dataYmal[.yaml]  configYaml[.yaml]    epochNo ")
    print("USAGE[predict]: python yolov8.py  predict  bestWeight       DataFolder/fileName  manual/auto  save/nosave")
    print("*"*80)
      
#=========================================================
# SHow OpenCV-style Image Window
#=========================================================
def show_image(image,windowTitle="",x=0,y=0,showMode="show") :
    global screenWidth, screenHeight

    if windowTitle == "" :
        windowTitle = "Result Image"
        
    resizedImage = cv2.resize(image, (screenWidth, screenHeight), interpolation=cv2.INTER_CUBIC)
    cv2.namedWindow(windowTitle)
    cv2.moveWindow(windowTitle,x,y)
    cv2.resizeWindow(windowTitle, screenWidth, screenHeight)
    
    cv2.imshow(windowTitle,resizedImage)
        
    if showMode == 'manual':
        miliSecond = 0
    else:
        miliSecond = 10

    inKey = cv2.waitKey(miliSecond)
    cv2.destroyAllWindows()
    
    return inKey

#=========================================================
# Get DataSet File(s)/Folder Info.
#=========================================================
def train_custom_data(dataYaml="yolov8_ocr_data.yaml", cfgYaml="yolov8_ocr_config.yaml",  epochNo = 40) :

    # For Training
    model = YOLO("yolov8m.yaml")    # build a new model from scratch
    model = YOLO("yolov8m.pt")     # load a pretrained model (recommended for training)
    
    model.train(data=dataYaml, cfg=cfgYaml, imgsz=640, batch=8,  epochs=epochNo, verbose=False, device=0)  # train the model
            #data="D:/myprojects/yolov8_ocr/ultralytics/cfg/datasets/yolov8_ocr.yaml"
            #cfg="D:/myprojects/yolov8_ocr/ultralytics/cfg/yolov8_ocr.yaml"
    gpu_status.displayGpuStatus(1)

    metrics = model.val()  # evaluate model performance on the validation set
    #path = model.export(format="onnx")  # export the model to ONNX format
    
    print("*"*80)
    print("Training Ended !!!")
    print("*"*80)
    
#=========================================================
# Get DataSet File adn Predict
#=========================================================
# def predict_data(bestWeight = "best.pt", dataFolder = "dataset_ocr/test/images/", 
def predict_data(bestWeight = "best.pt", dataFolder = "dataset_char/test/images/", 
                 showMode = 'manual', saveMode = 'save' ) :   
    cwd = os.getcwd()
    argFullPath =  os.path.join(cwd, dataFolder)
    imgDataList = []
    
    print("data_path: ",argFullPath)
    if os.path.isfile(argFullPath) :
        imgDataList.append(argFullPath)
    elif os.path.isdir(argFullPath) :
        fileList = os.listdir(argFullPath)
        imgDataList = [os.path.join(argFullPath,file) for file in fileList 
                        if (file.endswith(".jpg") or file.endswith(".png") or 
                            file.endswith(".JPG") or file.endswith(".PNG"))]
    #---------------------------------------------------------------------------------------------
    # Predice for all document images in a test folder
    #---------------------------------------------------------------------------------------------
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     
    model = YOLO("yolov8m.yaml")    # build a new model from scratch
    model = YOLO(bestWeight)     # load a pretrained model (recommended for training)
    gpu_status.displayGpuStatus(1)
    '''
    if device != "cpu" :
        print("!"*80)
        model = torch.nn.DataParallel(model)
        model.to(device)
    '''
    print("* Best Weight [%s] is Used. "  %bestWeight)
    
    originalExt = imgDataList[0][-4:].upper()
    
    if originalExt == '.JPG' :
        newExt = '.png'
    elif originalExt == '.PNG' :
        newExt = '.jpg'
    else :
        print("\n\n Image Format Mismatch !!! [.jpg or .png needed]")
        exit[1]
        
    imgSeq = 1    
    for imgFileName in imgDataList :
        results = model(imgFileName,verbose=False)                            # predict on an image
        
        resultImage = cv2.imread(imgFileName,cv2.IMREAD_COLOR)
        #----------------------------
        # Process for all found MBRs
        #----------------------------
        for result in results:
            boxes = result.boxes  # Boxes object(tensor) for bbox outputs
            '''
            #masks = result.masks  # Masks object for segmentation masks outputs
            #keypoints = result.keypoints  # Keypoints object for pose outputs
            #probs = result.probs  # Probs object for classification outputs
            '''
            # Get MBR positional Values
            mbrLeft =[];   mbrRight=[];   mbrTop=[]; mbrBottom=[]
            confidences = [];   classes = []                

            no=0
            for l,t,r,b in boxes.xyxy :
                # Get MBR values
                left = round(l.item()); right = round(r.item())
                top = round(t.item());  bottom = round(b.item())
                mbrLeft.append(left);  mbrTop.append(top);   
                mbrRight.append(right); mbrBottom.append(bottom)
                # Get Confidence Values
                confValue = round(boxes[no].conf.item(),2);   
                confidences.append(confValue)
                # Get Class Values
                clsValue = round(boxes[no].cls.item());   
                classes.append(clsValue)

                #print(left,right,top,bottom,clsValue)
                
                # Draw Rectangle
                #cv2.rectangle(resultImage,(left,top),(right,bottom),clsColor[clsValue], thickness=2)
                cv2.rectangle(resultImage,(left,top),(right,bottom),(0,255,0), thickness=2)

                '''
                # Write MBR Index
                cv2.putText(resultImage, str(no) , (mbrLeft[no]-10,mbrTop[no]+5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, cDarkGray)
                '''
                # Write Class Name
                cv2.putText(resultImage, clsType[clsValue] , (mbrLeft[no]-10,mbrTop[no]+5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, cDarkGray)
                
                # Write Confidence Value
                if confValue < 0.6:
                    colorConf = (0,0,255)
                else :
                    colorConf = cDarkGray
                cv2.putText(resultImage, str(confValue) , (mbrRight[no],mbrBottom[no]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.2, colorConf)
                
                no += 1

        if saveMode == 'save':
            resultFileName = imgFileName.replace(originalExt,newExt,1)
            #print("saveMode, resultFileName")
            cv2.imwrite(resultFileName,resultImage)
        else :
            resultFileName = ""
        
        if show_image(resultImage,resultFileName,0,0,showMode) == ord('q'):
            cv2.destroyAllWindows()
            break
        
        imgSeq += 1
        print("...... [%d] of %d predicted."%(imgSeq, len(imgDataList)), end='\r')
        
    print("\n","*"*80)
    print("Prediction Ended(Stopped) !!!")
    print("*"*80)

#=================================
# main
#=================================        
if __name__ == "__main__":
    # Check Parameters
    print(cv2.__version__)
    if len(sys.argv) > 3  :   
        get_screen_resolution()
        # Check if Training Mode
        if sys.argv[1] == 'train' :
            dataYaml = sys.argv[2]; configYaml=sys.argv[3]; epochNo = int(sys.argv[4])
            train_custom_data(dataYaml, configYaml, epochNo)   
        elif sys.argv[1] == 'predict' :
            bestWeight = sys.argv[2];   dataFolder = sys.argv[3];   
            showMode = sys.argv[4]; saveMode = sys.argv[5]
            predict_data(bestWeight, dataFolder, showMode, saveMode)   
        else :
            show_program_usage() 
    else :
        show_program_usage()            
    