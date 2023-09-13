
import cv2, os
import numpy as np
from random import randint
import random
from PIL import ImageFont, ImageDraw, Image

FONT_SIZE = 50
FONT_WIDTH = FONT_SIZE
FONT_HEIGHT = FONT_SIZE
MARGIN_WIDTH = 10
MARGIN_HEIGHT = 10

savePath = './generated_font_images'

if not os.path.isdir(savePath):
    os.makedirs(savePath)
font_size = FONT_SIZE

fontKorean = []
fontKorean.append(ImageFont.truetype("c://Windows/Fonts/gulim.ttc",font_size))
fontKorean.append(ImageFont.truetype("c://Windows/Fonts/H2MJRE.TTF",font_size)) #Hanyang Myungjo
fontKorean.append(ImageFont.truetype("c://Windows/Fonts/NanumGothic.ttf",font_size)) #Nanum Myungjo
fontEnglish=[]                            
fontEnglish.append(ImageFont.truetype("c://Windows/Fonts/times.ttf",font_size)) # English
#fontEnglish.append(ImageFont.truetype("c://Windows/Fonts/sserife.fon",font_size)) # English
#배경 생성
def make_background_img(w,h, bgColor=(255, 255, 255)):
    background_size=w * h * 3
    img = np.full((h+30,w+20,3),  bgColor, np.uint8)
    img = Image.fromarray(img)
    img = np.array(img)
    
    return img

# make char image
def make_char_font_image(charToMake):

    w = len(charToMake) * font_size +100
    h = font_size*3
    #255 로 배경생성(높이, 폭, 채널) 색상값
    img = np.full((h, w, 3), (255, 255, 255), np.uint8)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    text = charToMake
    
    #use font depending on country charset
    if ord(charToMake) >= int(0xac00) :            # Korean Char
        fontName = fontKorean[random.randint(0,len(fontKorean)-1)]
    else :
        fontName = fontEnglish[random.randint(0,len(fontEnglish)-1)]
    
    draw.text((20, 20),  text, font=fontName, fill=(0, 0, 0))
        
    img = np.array(img)

    return img

#가장자리 자르기 https://www.javaer101.com/ko/article/1052221.html

def cropImage(img):
    rsz_img = cv2.resize(img, None, fx=1.0, fy=1.0) # resize since image is huge
    gray = cv2.cvtColor(rsz_img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY)

    # find where the signature is and make a cropped region
    points = np.argwhere(thresh_gray==0) # find where the black pixels are
    points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
    x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
    #x, y, w, h = x-2, y-2, w+2, h+2 # make the box a little bigger
    

    crop =  np.full((h, w, 3), (255, 255, 255), np.uint8)

    crop =img[y:y+h, x:x+w] # create a cropped region of the gray image
    
    return crop


def generate_labe_file(labelFile, charNoPerLine = 50, lineNoPerPage = 40, totalPageNo = 10):

    totalCharNo = lineNoPerPage * charNoPerLine

    # korean 11172 chars
    koreanCharList = [];    numberCharList = [];
    englishCharList = [];   englishUpperCharList = [];   
    keyboardCharList1 = [];keyboardCharList2 = [];keyboardCharList3 = [];keyboardCharList4 = [];
    keyboardCharList = []
    
    for i in range(0xac00,0xd7a4):
        koreanCharList.append(chr(i))
    print(koreanCharList[:3],"......",koreanCharList[-3:])        
    
    # 10 Numbers
    for i in range(0x30,0x39):
        numberCharList.append(chr(i))
    # English Upper
    for i in range(0x41,0x5A):
        englishUpperCharList.append(chr(i))
        englishCharList.append(chr(i))
    for i in range(0x61,0x7A):
        englishCharList.append(chr(i))
    print(englishCharList[:3],"......",englishCharList[-3:])        

    # Keboard Chars
    for i in range(0x21,0x2F):
        keyboardCharList1.append(chr(i))
    for i in range(0x3A,0x40):
        keyboardCharList2.append(chr(i))
    for i in range(0x5B,0x60):
        keyboardCharList3.append(chr(i))
    for i in range(0x7B,0x7E):
        keyboardCharList4.append(chr(i))
    #print(keyboardCharList[:3],"......",keyboardCharList[-3:])        

    #Choose 200 Korean characters randomly
    #selectedKoreanCharList = random.sample(koreanCharList,500)
    keyboardCharList = keyboardCharList1 + keyboardCharList2 + keyboardCharList3 + keyboardCharList4 
    #allCharList = keyboardCharList + numberCharList + englishCharList + koreanCharList 
    allCharList = numberCharList + englishCharList + koreanCharList 
    #lastNdxInAllCharList = len(allCharList) -1 
    lastNdxInAllCharList = 199
    
    charNdx = 0
    
    for page in range(totalPageNo):
        
        # make a carh image and get its size
        charImg = make_char_font_image('활')
        charCard = cropImage(charImg)
        #cv2.imshow("Font",charCard)
        #cv2.imshow("Image",charImg)
        
        charHeight, charWidth, channels = charCard.shape
        
        bgImgWidth = (MARGIN_WIDTH+charWidth)*(charNoPerLine+3) + MARGIN_WIDTH
        bgImgHeight = (MARGIN_HEIGHT+charWidth)*(lineNoPerPage+3) + MARGIN_HEIGHT
        bgImg = make_background_img(bgImgWidth,bgImgHeight, (255, 255, 255))
        bgImgHeight, bgImgWidth, _ = bgImg.shape
        displayImg = bgImg.copy()
        
        allLabelStr = [];   allCharStr = []
        
        seqNo = "%05d" % page
        # Open a file for Yolo data Txt File
        pageLabelFile = labelFile + '-' + str(seqNo) + '.txt'
        labelTxt = open(savePath+"/"+pageLabelFile,"w")

        # Open a file for Generated Char Txt File
        pageCharFile = labelFile + "_genchar" + '-' + str(seqNo)  +  '.txt'
        genCharTxt = open(savePath+"/"+pageCharFile,"w")

        for lineNo in range(lineNoPerPage):
            if lineNo == 0:
                prevBottom = 0
                maxPosBottom = 0
            for charNo in range(charNoPerLine):
                #character = allCharList[charNdx]
                arrayNdx = random.randint(0,lastNdxInAllCharList)
                character = allCharList[arrayNdx]
                
                '''
                if arrayNdx < len(numberCharList) :
                    labelCls = 0
                elif arrayNdx < len(englishCharList) :
                    labelCls = 1
                else :
                    labelCls = 2
                '''
                labelCls = arrayNdx
                charImg = make_char_font_image(character)
                '''
                cv2.imshow("Img",charImg)
                cv2.moveWindow("Img",0,0)
                '''
                charCard = cropImage(charImg)
                
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
                
                charStr = "%s " %character
                allCharStr.append(charStr)
                
                if cv2.waitKey(5) == ord('q') :
                    exit(0)

                prevPosRight = posRight
                
                if posBottom > maxPosBottom:
                    maxPosBottom = posBottom
                charNdx += 1
                print("......\t\t (%d of %d) done"%(charNdx,totalCharNo*totalPageNo),end="\r")


            allCharStr.append('\n')
            prevBottom = maxPosBottom
            
        # Write Data for Yolo Training
        for line in allLabelStr:
            labelTxt.write(line)
        labelTxt.close()
        
        # Write Generated char for each page
        for line in allCharStr:
            genCharTxt.write(line)
        genCharTxt.close()
    
        cv2.imshow("FontImage",displayImg)
        labelImgFile = pageLabelFile.replace(".txt",".jpg")
        cv2.imwrite(savePath+"/"+labelImgFile,bgImg)
        
    print("\nFinished......Label Text File: %s" %labelFile)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    generate_labe_file("fontLabel", 10, 10, 30)
    
    
    
    