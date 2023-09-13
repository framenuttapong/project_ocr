#******************************************************************************
# Detect a Character as one of Keyboard Characters or Korean Character Class 
#******************************************************************************
import cv2, os,sys
import numpy as np
from random import randint
import random
from PIL import ImageFont, ImageDraw, Image

FONT_SIZE = 30
FONT_WIDTH = FONT_SIZE
FONT_HEIGHT = FONT_SIZE
MARGIN_WIDTH = 10
MARGIN_HEIGHT = 10

savePathImage = './generated_font_images_all/image'
savePathLabel = './generated_font_images_all/label'
savePathData = './generated_font_images_all/data'

fontPathKorean = 'Fonts4Train/Korean/'
fontPathEnglish = 'Fonts4Train/English/'
fontPathThai = 'Fonts4Train/Thai/'

if not os.path.isdir(savePathImage):
    os.makedirs(savePathImage)
if not os.path.isdir(savePathLabel):
    os.makedirs(savePathLabel)
if not os.path.isdir(savePathData):
    os.makedirs(savePathData)
    
if not os.path.isdir(fontPathKorean):
    os.makedirs(fontPathKorean)
if not os.path.isdir(fontPathEnglish):
    os.makedirs(fontPathEnglish)
    
cwd = os.getcwd()

# Get Korean Font File List
fontDirKorean = cwd+"/"+fontPathKorean
fontKorean = []
fileList = os.listdir(fontDirKorean)
#fontKoreanList = [os.path.join(fontDir,file) for file in fileList 
fontKoreanList = [file for file in fileList 
                if file.endswith(".ttf") or file.endswith(".TTF") or 
                    file.endswith(".ttc") or file.endswith(".TTC") or 
                    file.endswith(".otf") or file.endswith(".OTF") 
                ]

# Get English Font File List
fontDirEnglish = cwd+"/"+fontPathEnglish
fontEnglish=[]                            
fileList = os.listdir(fontDirEnglish)
#fontEnglishList = [os.path.join(fontDir,file) for file in fileList 
fontEnglishList = [file for file in fileList 
                if file.endswith(".ttf") or file.endswith(".TTF") or 
                    file.endswith(".ttc") or file.endswith(".TTC") or 
                    file.endswith(".otf") or file.endswith(".OTF") 
                ]

fontDirThai = cwd+"/"+fontPathThai
fontThai = []
fileList = os.listdir(fontDirThai)
# fontThaiList = [os.path.join(fontDir,file) for file in fileList
fontThaiList = [file for file in fileList 
                if file.endswith(".ttf") or file.endswith(".TTF") or 
                    file.endswith(".ttc") or file.endswith(".TTC") or 
                    file.endswith(".otf") or file.endswith(".OTF") 
                ]

#print(fontKoreanList)
#print(fontEnglishList)
#print(fontThaiList)

# fontAllList = fontEnglishList + fontKoreanList
fontAllList = fontEnglishList + fontThaiList

# make fonts lists
# for fontName in fontKoreanList:
#     fontPath = os.path.join(fontDirKorean,fontName)
#     if os.path.exists(fontPath) :
#         #print(fontPath)
#         if not fontPath.endswith('.fon') :
#             fontKorean.append(ImageFont.truetype(fontPath,FONT_SIZE))

for fontName in fontEnglishList:
    fontPath = os.path.join(fontDirEnglish,fontName)
    if os.path.exists(fontPath) :   
        #print(fontPath)
        if not fontPath.endswith('.fon') :
            fontEnglish.append(ImageFont.truetype(fontPath,FONT_SIZE))

for fontName in fontThaiList:
    fontPath = os.path.join(fontDirThai,fontName)
    if os.path.exists(fontPath) :
        #print(fontPath)
        if not fontPath.endswith('.fon') :
            fontThai.append(ImageFont.truetype(fontPath,FONT_SIZE))

# totalFontFileNo = len(fontKorean) + len(fontEnglish)
totalFontFileNo = len(fontThai) + len(fontEnglish)

print("total # of Font Files to Train: ",totalFontFileNo)
# Special characters to add space when making font images
specialUpperCharList = [chr(34), chr(39), chr(94), chr(96)]
specialLowerCharList = [chr(44), chr(46), chr(95)]
specialMiddleCharList = [chr(45), chr(126)]

#=========================================================================================
# make a font background image
def make_background_img(w,h, bgColor=(255, 255, 255)):
    background_size=w * h * 3
    img = np.full((h+50,w+50,3),  bgColor, np.uint8)
    img = Image.fromarray(img)
    img = np.array(img)
    
    return img

# make char image
def make_char_font_image(charToMake, fontCountry, fontNo):

    w = len(charToMake) * FONT_SIZE + 50
    h = FONT_SIZE*3
    #255 로 배경생성(높이, 폭, 채널) 색상값
    img = np.full((h, w, 3), (255, 255, 255), np.uint8)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    text = charToMake
    
    #print("charToMake: ",charToMake, " fontCountry: ", fontCountry, " fontNo: ", fontNo)
    # if fontCountry == 'KorFont' :            # Korean CharfontAllList
    #     draw.text((20, 20),  text, font=fontKorean[fontNo], fill=(0, 0, 0))
    if fontCountry == 'ThaFont' :            # Thai CharfontAllList
        draw.text((20, 20),  text, font=fontThai[fontNo], fill=(0, 0, 0))
    else :
        draw.text((20, 20),  text, font=fontEnglish[fontNo], fill=(0, 0, 0))

    
        
    img = np.array(img)

    rsz_img =  img

    #가장자리 자르기 https://www.javaer101.com/ko/article/1052221.html

    #rsz_img = cv2.resize(img, None, fx=1.0, fy=1.0) # resize since image is huge
    gray = cv2.cvtColor(rsz_img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY)

    # find where the signature is and make a cropped region
    points = np.argwhere(thresh_gray==0) # find where the black pixels are
    points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
    x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
    #x, y, w, h = x-2, y-2, w+2, h+2 # make the box a little bigger
    
    crop =  np.full((h, w, 3), (255, 255, 255), np.uint8)

    crop =img[y:y+h, x:x+w] # create a cropped region of the gray image
    
    # add proper space for special upper characters such as "(34)  '(39)   ^(94)  `(96)
    if charToMake in specialUpperCharList :
        h, w, c = crop.shape
        if charToMake == '^' :
            newH = h*2
        else :            
            newH = h*3
        newImg = img = np.full((newH,w,3),(255,255,255))
        newImg = np.array(newImg)
        newImg[:h,:] = crop[:,:]
        return newImg
    # add proper space for special middle characters such as -(45) ~(126)
    elif charToMake in specialMiddleCharList :
        h, w, c = crop.shape
        if charToMake == '_' :
            newH = h*8
        else :
            newH = h*6
        newImg = img = np.full((newH,w,3),(255,255,255))
        newImg = np.array(newImg)
        newImg[int(newH/2):int(newH/2)+h,:] = crop[:,:]
        return newImg
    # add proper space for special lower characters such as ,(44)  .(46)   _(95)
    elif charToMake in specialLowerCharList :
        h, w, c = crop.shape
        if charToMake == '_' :
            newH = h*8
        elif charToMake == ',' :
            newH = h*3
        else :
            newH = h*5
        newImg = img = np.full((newH,w,3),(255,255,255))
        newImg = np.array(newImg)
        newImg[newH-h:,:] = crop[:,:]
        return newImg
        
    return crop


def generate_label_file(labelFile, charNoPerLine = 50, lineNoPerPage = 40):

    FINISH_NUM = 1000000000
    totalCharNo = lineNoPerPage * charNoPerLine

    # korean 11172 chars
    # koreanCharList = [];    
    numberCharList = [];
    englishLowerCharList = [];   englishUpperCharList = [];
    keyboardCharList = []
    keyboardCharList1 = [];keyboardCharList2 = [];keyboardCharList3 = [];keyboardCharList4 = [];
    allCharList = [] 

    # thai 
    thaiConsonantCharList = [];
    thaiVowelCharList = []; thaiSignCharList = [];
    thaiDigitCharList = []; thaiToneMarkCharList = [];

    # Thai type 1
    for i in range(0x0E01,0x0E2E+1): # Consonants 
        thaiConsonantCharList.append(chr(i))
        allCharList.append(chr(i))

    thaiSignCharList.append(chr(0x0E2F)) # Sign ฯ
    allCharList.append(chr(0x0E2F))

    for i in range(0x0E30,0x0E3A+1): # Vowels
        thaiVowelCharList.append(chr(i))
        allCharList.append(chr(i))
    
    # thaiCharList.append(chr(0x0E3F)) # ฿

    for i in range(0x0E40,0x0E47+1): # Vowels
        thaiVowelCharList.append(chr(i))
        allCharList.append(chr(i))

    for i in range(0x0E48,0x0E4B+1): # Tone Marks
        thaiToneMarkCharList.append(chr(i))
        allCharList.append(chr(i))

    for i in range(0x0E4C,0x0E4F+1): # Signs
        thaiSignCharList.append(chr(i))
        allCharList.append(chr(i))

    for i in range(0x0E50,0x0E59+1): # ThaiDigits ๐๑๒๓๔๕๗๘๙
        thaiDigitCharList.append(chr(i))
        allCharList.append(chr(i))

    for i in range(0x0E5A,0x0E5B+1): # Signs
        thaiSignCharList.append(chr(i))
        allCharList.append(chr(i))

    keyboardCharList.append(chr(0x0E3F)) # ฿
    allCharList.append(chr(0x0E3F)) 

    #print(keyboardCharList[:3],"......",keyboardCharList[-3:])


    # Null label
    for i in range(0x0021):
        allCharList.append('')
    
    # ASCII Symbols '!' ~ '/'
    for i in range(0x21,0x30):
        keyboardCharList1.append(chr(i))
        allCharList.append(chr(i))

    # 10 Numbers '0' ~ '9'
    for i in range(0x30,0x3A):
        numberCharList.append(chr(i))
        allCharList.append(chr(i))

    # ASCII Symbols ':' ~ '@'
    for i in range(0x3A,0x41):
        keyboardCharList2.append(chr(i))
        allCharList.append(chr(i))

    # English Upper 'A' ~ 'Z'
    for i in range(0x41,0x5B):
        englishUpperCharList.append(chr(i))
        allCharList.append(chr(i))

    # ASCII Symbols '[' ~ '''
    for i in range(0x5B,0x61):
        keyboardCharList3.append(chr(i))
        allCharList.append(chr(i))

    # English Lower 'a' ~ 'z'
    for i in range(0x61,0x7B):
        englishLowerCharList.append(chr(i))
        allCharList.append(chr(i))

    # ASCII Symbols '{' ~ '~'
    for i in range(0x7B,0x7F):
        keyboardCharList4.append(chr(i))
        allCharList.append(chr(i))

    
    # lastNdxNonKoreanCharList = len(allCharList)
    lastNdxNonThaiCharList = len(allCharList)

    # # Korean 11,172 characters    
    # for i in range(0xac00,0xd7a4):
    #     koreanCharList.append(chr(i))
    #     allCharList.append(chr(i))
        
    #print("numberCharList[%d]: " % len(numberCharList),numberCharList)
    keyboardCharList = keyboardCharList1 + keyboardCharList2 + keyboardCharList3 + keyboardCharList4 
    #print("keboardCharList[%d]: "  % len(keyboardCharList),keyboardCharList)
    #print("englishUpperCharList[%d]: " % len(englishUpperCharList), englishUpperCharList)
    #print("englishLowerCharList[%d]: " % len(englishLowerCharList),englishLowerCharList)        
    
    print("total Char Label # : ", lastNdxNonThaiCharList,"\n")
    
    lastNdxAllCharList = len(allCharList) 
    #print("lastNdxAllCharList: ",lastNdxAllCharList)
   
    lastProcessedCharNdx = 0
    for fontSeq in range(len(fontAllList))    :
        print("\nfont: ",fontAllList[fontSeq])
        
        fontName = fontAllList[fontSeq]

        charNdx = 33 # 0x0021
    
        #for page in range(totalPageNo):
        page = 0
        while True:
            if charNdx >= FINISH_NUM :
                break
            
            # make a carh image and get its size
            #charImg = make_char_font_image('활', 'KorFont', 0)
            #charCard = cropImage(charImg)
            
            charCard = make_char_font_image('ก', 'ThaFont', 0)
            #cv2.imshow("Font",charCard)
            #cv2.imshow("Image",charImg)
            
            charHeight, charWidth, channels = charCard.shape
            
            bgImgWidth = (MARGIN_WIDTH+FONT_SIZE)*(charNoPerLine+2)
            bgImgHeight = (MARGIN_HEIGHT+FONT_SIZE)*(lineNoPerPage+2)

            bgImg = make_background_img(bgImgWidth,bgImgHeight, (255, 255, 255))
            bgImgHeight, bgImgWidth, _ = bgImg.shape
            
            displayImg = bgImg.copy()
            
            allLabelStr = [];   allCharStr = []
            
            seqNo = "%05d" % page
            # Open a file for Yolo data Txt File
            pageLabelFile = labelFile + '-' + str(seqNo) + '-' + fontName + '.txt'
            labelTxt = open(savePathLabel+"/"+pageLabelFile,"w")

            # Open a file for Generated Char Txt File
            pageCharFile = labelFile + "_genchar" + '-' + fontName + str(seqNo)  +  '.txt'
            genCharTxt = open(savePathData+"/"+pageCharFile,"w",encoding='utf-8')

            for lineNo in range(lineNoPerPage):
                if charNdx >= FINISH_NUM :
                    break
                if lineNo == 0:
                    prevBottom = 0
                    maxPosBottom = 0
                for charNo in range(charNoPerLine):
                    if charNdx >= FINISH_NUM :
                        break
                    #character = allCharList[charNdx]
                    #print("charNdx: ",charNdx)
                    #print("arrayNdx: ",allCharList[charNdx])
                    #arrayNdx = ord(allCharList[charNdx])
                    arrayNdx = charNdx

                    character = allCharList[arrayNdx]
                    
                    # if arrayNdx >= lastNdxNonKoreanCharList:
                    #     labelCls = lastNdxNonKoreanCharList
                    if arrayNdx >= lastNdxNonThaiCharList:
                        labelCls = lastNdxNonThaiCharList
                    else :
                        labelCls = arrayNdx
                    #print("labelCls...\t",labelCls)

                    if fontSeq < len(fontEnglishList) :
                        charCard = make_char_font_image(character, 'EngFont', fontSeq)
                    else :
                        charCard = make_char_font_image(character, 'ThaFont', fontSeq-len(fontEnglish))
                    '''
                    cv2.imshow("Img",charImg)
                    cv2.moveWindow("Img",0,0)
                    '''
                    #charCard = cropImage(charImg)
                    
                    '''
                    cv2.imshow("Card",charCard)
                    cv2.moveWindow("Card",100,100)
                    if cv2.waitKey(0) == ord('q') :
                        exit(0)
                    '''
                        
                    charHeight, charWidth, channels = charCard.shape
                    #print("charCard shape: ",charCard.shape)
                    
                    if charNo == 0:
                        prevPosRight = 0
                        
                    if randint(0,1) == 1   :
                        posLeft = prevPosRight + MARGIN_WIDTH - randint(1,3)
                    else :
                        posLeft = prevPosRight + randint(1,3)              # print justright of previous char image
                        
                    posTop = prevBottom + MARGIN_HEIGHT 
                    posRight = posLeft + charWidth
                    posBottom = posTop + charHeight
                    
                    #print(posLeft,posTop,charWidth,charHeight,"\n")
                    #bgImg[posTop:posBottom, posLeft:posRight] = charCard[:charHeight,:charWidth]
                    bgImg[posTop:posBottom, posLeft:posRight] = charCard[:charHeight,:charWidth]
                    #print("posLeft: ",posLeft," posRight: ",posRight)
                    displayImg[posTop:posBottom, posLeft:posRight] = charCard[:charHeight,:charWidth]
                    cv2.rectangle(displayImg, (posLeft,posTop), (posRight,posBottom), (0,0,255), 1)

                    boxCx = ((float(posLeft+posRight))/2.0) / float(bgImgWidth)
                    boxCy = ((float(posTop+posBottom))/2.0) / float(bgImgHeight)
                    boxWidth = float(abs(posRight-posLeft)) / float(bgImgWidth)
                    boxHeight = float(abs(posBottom-posTop)) / float(bgImgHeight)

                    labelStr = "%d %f %f %f %f\n" %(labelCls, boxCx, boxCy, boxWidth, boxHeight)
                    
                    allLabelStr.append(labelStr)
                    
                    charStr = "%s " % character
                    allCharStr.append(charStr)
                    
                    if cv2.waitKey(5) == ord('q') :
                        exit(0)

                    prevPosRight = posRight
                    
                    if posBottom > maxPosBottom:
                        maxPosBottom = posBottom
                    '''
                    if fontSeq < len(fontEnglishList) :
                        charNdx += 1
                    else :
                        if charNdx >= lastNdxNonKoreanCharList:
                            charNdx += random.randint(5,15)
                        else :
                            charNdx += 1
                    '''    
                    charNdx += 1
                    lastProcessedCharNdx += 1
                    
                    if fontSeq < len(fontEnglish) :                 # non Korean Font
                        # if charNdx >= lastNdxNonKoreanCharList :
                        if charNdx >= lastNdxNonThaiCharList : # non Thai Font
                            charNdx = FINISH_NUM
                            break
                    else :                                          # Thai Font
                        if charNdx >= len(allCharList)-1 :
                            charNdx = FINISH_NUM
                            break

                allCharStr.append('\n')
                prevBottom = maxPosBottom
                
                print("......\t\t\t (%10d char images,\t%d of %d font-files) done"
                      %(lastProcessedCharNdx, fontSeq, len(fontAllList)),
                      end="\r")
                
            # Write Data for Yolo Training
            for line in allLabelStr:
                labelTxt.write(line)
            labelTxt.close()
            
            # Write Generated char for each page
            for line in allCharStr:
                genCharTxt.write(line)
            genCharTxt.close()
        
            cv2.imshow("fontImage",displayImg)
            #if charNdx < FINISH_NUM :
            labelImgFile = pageLabelFile.replace(".txt",".jpg")
            cv2.imwrite(savePathImage+"/"+labelImgFile,bgImg)
            
            page += 1
        #cv2.waitKey(0)    
    print("\nFinished......Label Text File: %s" %labelFile)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    generate_label_file("font", 10, 10)
    
    