import os
import cv2
import pprint
import aiohttp
import requests
from enum import Enum
from io import BytesIO
from typing import List, Optional
from urllib.parse import urlparse



from fastapi.responses import HTMLResponse

import io
import PIL
from PIL import Image, ImageEnhance

import numpy as np
from PIL import Image, ImageSequence
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

# import logfile
# from logfile import #logger


SQUARE_YARDS_LOGO = Image.open('api/slogo.png')
IC_LOGO = Image.open('api/iclogo2.png')
POSI_LIST = ["centre", "bottom_left", "bottom_right", "bottom"]

app = FastAPI(
    title="sqy-watermark-engine",
    description="Use this API to paste Square Yards logo as a watermark at the center of input images",
    version="1.0",
)

@app.get("/")
async def root():
    return "Hello World!!!"

class URL(BaseModel):
    url_: str
 
 
class ImageDetails(BaseModel):
    url_: str
    width_percentage: Optional[float] = 0.2
    position: Optional[str] = "centre"

sample_list_for_without_exten=[]

def sample_list_ext(L):
    sample_list_for_without_exten.append(L)
    #print(sample_list_for_without_exten)

total_request_extension=[]

def total_req_ext(k):
    total_request_extension.append(k)
    #print(total_request_extension)


sample_list_for_without_exten2=[]

def sample_list_ext2(L):
    sample_list_for_without_exten2.append(L)
    #print(sample_list_for_without_exten2)

total_request_extension2=[]

def total_req_ext2(k):
    total_request_extension2.append(k)
    #print(total_request_extension2)


sample_list_for_logo_enhancement=[]

def sample_list_logo_enhancement(L):
    sample_list_for_logo_enhancement.append(L)
    #print(sample_list_for_logo_enhancement)

total_request_logo_enhancement=[]

def total_req_logo_enhancement(k):
    total_request_logo_enhancement.append(k)
    #print(total_request_logo_enhancement)

def extract_filename(URL):
    parsed = urlparse(URL)
    return os.path.basename(parsed.path)
 
 
async def get_image_properties(URL, width_percentage=None, position=None):
    filename = None
    try:
        filename = extract_filename(URL)
        filename = filename.strip()
    except Exception as e:
        print(e)
        ##logger.info("Error: HTTPException(status_code=406, detail=Not a valid URL)")
        raise HTTPException(status_code=406, detail="Not a valid URL")
 
    if URL.lower().endswith((".jpg", ".png", ".jpeg", ".gif", ".webp")) == False:
        ##logger.info("Error: HTTPException(status_code=406, detail=Not a valid URL)")
        raise HTTPException(status_code=406, detail="Not a valid URL")
 
    if width_percentage and width_percentage > 1:
        ##logger.info("Error: HTTPException(status_code=406, detail=Please chose the value of width_percentage between 0.01 and 1.0)")
        raise HTTPException(status_code=406, detail="Please chose the value of width_percentage between 0.01 and 1.0")
 
    if position and position not in POSI_LIST:
        ##.info("Error: HTTPException(status_code=406, detail=Please chose a value of position from")
        raise HTTPException(status_code=406, detail="Please chose a value of position from: " + ", ".join(POSI_LIST))
 
   
    contents = None
    original_image = None
    '''
    try:
        # contents = requests.get(URL, timeout=10).content
        print(URL)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(URL) as resp:
                print(URL)
                contents = await resp.read()
 
        if contents == None:
            #logger.info("Error: HTTPException(status_code=406, detail=No image found.")
            raise HTTPException(status_code=406, detail="No image found.")
 
        original_image = Image.open(BytesIO(contents))


        

    except Exception as e:
        print(e)
        #logger.info("Error: HTTPException(status_code=400, detail= Error while reading the image. Make sure that the URL is a correct image link.")
        raise HTTPException(status_code=400, detail="Error while reading the image. Make sure that the URL is a correct image link.")
    
    print(filename, original_image)
    '''
    return filename, original_image
 
 
def paste_logo(original_image, width_percentage, logo, position="centre"):

    try:
        logo_width = int(original_image.size[0]*width_percentage)
        logo_height = int(logo.size[1]*(logo_width/logo.size[0]))
 
        if logo_height > original_image.size[1]:
            logo_height = original_image.size[1]
    
        if position == "centre":
            logo = logo.resize((logo_width, logo_height))
 
            top = (original_image.size[1]//2) - (logo_height//2)
            left = (original_image.size[0]//2) - (logo_width//2)
            original_image.paste(logo, (left, top), mask=logo)
 
        elif position == "bottom_right":
            logo = logo.resize((logo_width, logo_height))
 
            top = original_image.size[1] - logo_height
            left = original_image.size[0] - logo_width
            original_image.paste(logo, (left, top), mask=logo)
 
        elif position == "bottom_left":
            logo = logo.resize((logo_width, logo_height))
 
            top = original_image.size[1] - logo_height
            left = 0
            original_image.paste(logo, (left, top), mask=logo)
    
        elif position == "bottom":
            logo = logo.resize((logo_width, logo_height))
 
            top = original_image.size[1] - logo_height
            left = (original_image.size[0]//2) - (logo_width//2)
            original_image.paste(logo, (left, top), mask=logo)
        #logger.info("logo added successfully")
        return original_image
    
    except Exception as e:
        print(e)
        #logger.info("logo adding unsuccessful")
        pass 
 
def get_format(filename):
    format_ = filename.split(".")[-1]
    if format_.lower() == "jpg":
        format_ = "jpeg"
    elif format_.lower == "webp":
        format_ = "WebP"
 
    return format_
 
 
def get_content_type(format_):
    type_ = "image/jpeg"
    if format_ == "gif":
        type_ = "image/gif"
    elif format_ == "webp":
        type_ = "image/webp"
    elif format_ == "png":
        type_ = "image/png"

    return type_
 
 
def get_final_image(image_details, original_image, width_percentage, logo, position, filename):

    try:
        original_image = paste_logo(original_image, width_percentage, logo, position)
        format_ = get_format(filename)
        quality = 90
        #logger.info("compression successful")
        return original_image, format_, quality

    except Exception as e:
        print(e)
        #logger.info("compression unsuccessful")    
        pass


async def get_body(URL):

    #print(Enhance_image)
    response = requests.get(URL)
    image_bytes = io.BytesIO(response.content)
    #print(image_bytes)
    image = PIL.Image.open(image_bytes)
    #print(image)
    filename = URL
    #print(filename)
    #this function get the format type of input image

    try:
        def get_format(filename):
            format_ = filename.split(".")[-1]
            if format_.lower() == "jpg":
                format_ = "jpeg"
            elif format_.lower == "webp":
                format_ = "WebP"
    
            return format_

            #this function for gave the same type of format to output
        def get_content_type(format_):
            type_ = "image/jpeg"
            if format_ == "gif":
                type_ = "image/gif"
            elif format_ == "webp":
                type_ = "image/webp"
            elif format_ == "png":
                type_ = "image/png"
            #print(type_)
            return type_

        format_ = get_format(filename)#here format_ store the type of image by filename    
   

        #This function calculate the brightness of input image 
        def calculate_brightness(image):
            greyscale_image = image.convert('L')
            histogram = greyscale_image.histogram()
            pixels = sum(histogram)
            brightness = scale = len(histogram)

            for index in range(0, scale):
                ratio = histogram[index] / pixels
                brightness += ratio * (-scale + index)

            return 1 if brightness == 255 else brightness / scale
    
        print("starting b=",calculate_brightness(image))#here print the float(calculate brightnes nummber)
    

        #______Here apply the Brightness and Color on image automatically according to there condition_____
        # if (calculate_brightness(image) > 0.6 and calculate_brightness(image) < 0.64 ): 
        #     enhancer_bright = ImageEnhance.Brightness(image)
        #     image = enhancer_bright.enhance(1.1)
        #     #print("bright 6")
        #     if image:
        #         enhancer_colors = ImageEnhance.Color(image)
        #         image = enhancer_colors.enhance(1.2)
        #         #print("color 6")
                  
        if (calculate_brightness(image) > 0.5 and calculate_brightness(image) < 0.55): 
            enhancer_bright = ImageEnhance.Brightness(image)
            image = enhancer_bright.enhance(1.1)
            #print("bright 5")
            if image:
                enhancer_colors = ImageEnhance.Color(image)
                image = enhancer_colors.enhance(1.3)
                #print("color 5")

        
        if (calculate_brightness(image)  > 0.4 and calculate_brightness(image) < 0.5 ):
            enhancer_bright = ImageEnhance.Brightness(image)
            image = enhancer_bright.enhance(1.2)
            #print("bright 4")
            if image:
                enhancer_colors = ImageEnhance.Color(image)
                image = enhancer_colors.enhance(1.3)
                #print("color 4")

        
        if (calculate_brightness(image)  > 0.3 and calculate_brightness(image) < 0.4):
            enhancer_bright = ImageEnhance.Brightness(image)
            image = enhancer_bright.enhance(1.3)
            #print("bright 3")
            if image:
                enhancer_colors = ImageEnhance.Color(image)
                image = enhancer_colors.enhance(1.4)
                #print("color 3")

        
        if (calculate_brightness(image)  > 0.2 and calculate_brightness(image) < 0.3 ):
            enhancer_bright = ImageEnhance.Brightness(image)
            image = enhancer_bright.enhance(1.4)
            #print("bright 2")
            if image:
                enhancer_colors = ImageEnhance.Color(image)
                image = enhancer_colors.enhance(1.4)
                #print("color 2")

        if (calculate_brightness(image)  > 0.1 and calculate_brightness(image) < 0.2):
            enhancer_bright = ImageEnhance.Brightness(image)
            image = enhancer_bright.enhance(1.5)
            #print("bright 1")
            if image:
                enhancer_colors = ImageEnhance.Color(image)
                image = enhancer_colors.enhance(1.3)
                #print("color 1")

        if (calculate_brightness(image)  > 0.001 and calculate_brightness(image) < 0.1 ):
            enhancer_bright = ImageEnhance.Brightness(image)
            image = enhancer_bright.enhance(2.0)
            #print("bright 001")
            if image:
                enhancer_colors = ImageEnhance.Color(image)
                image = enhancer_colors.enhance(1.6)
                print("color 001")
        
        if calculate_brightness(image) > 0.58: 
            enhancer_bright = ImageEnhance.Brightness(image)
            image = enhancer_bright.enhance(0.9)

        def calculate_brightness(image):
            greyscale_image = image.convert('L')
            histogram = greyscale_image.histogram()
            pixels = sum(histogram)
            brightness = scale = len(histogram)

            for index in range(0, scale):
                ratio = histogram[index] / pixels
                brightness += ratio * (-scale + index)

            return 1 if brightness == 255 else brightness / scale
    
        print("after",calculate_brightness(image))

           
        def calculate_brightness(image):
            greyscale_image = image.convert('L')
            histogram = greyscale_image.histogram()
            pixels = sum(histogram)
            brightness = scale = len(histogram)

            for index in range(0, scale):
                ratio = histogram[index] / pixels
                brightness += ratio * (-scale + index)

            return 1 if brightness == 255 else brightness / scale
        
        print("after enhn b=",calculate_brightness(image))
        image = PIL.Image.open(image_bytes)
        format_1 =image.format
        image.save("original_img."+format_1)
        img_s = cv2.imread("original_img."+format_1)
        img_hsv = cv2.cvtColor(img_s, cv2.COLOR_BGR2HSV)
        saturation = img_hsv[:, :, 1].mean()
        print("original saturation",saturation)
        
        if saturation >100 and saturation < 120 :
            enhancer_colors = ImageEnhance.Color(image)
            image = enhancer_colors.enhance(0.8)
            # if image:
            #     enhancer_colors = ImageEnhance.Contrast(image)
            #     image = enhancer_colors.enhance(0.9)
            #print("color 6")
            #print("color 6")
            image.save("original_img."+format_1)
            img_s = cv2.imread("original_img."+format_1)
            img_hsv = cv2.cvtColor(img_s, cv2.COLOR_BGR2HSV)
            saturation = img_hsv[:, :, 1].mean()
            print("after apply 100 =",saturation)
        
        if saturation >120 and saturation < 150 :
            enhancer_colors = ImageEnhance.Color(image)
            image = enhancer_colors.enhance(0.75)
            # if image:
            #     enhancer_colors = ImageEnhance.Contrast(image)
            #     image = enhancer_colors.enhance(0.9)
            #print("color 6")
            #print("color 6")
            image.save("original_img."+format_1)
            img_s = cv2.imread("original_img."+format_1)
            img_hsv = cv2.cvtColor(img_s, cv2.COLOR_BGR2HSV)
            saturation = img_hsv[:, :, 1].mean()
            print("after apply 120=",saturation)


        if saturation >150 :
            enhancer_colors = ImageEnhance.Color(image)
            image = enhancer_colors.enhance(0.6)
            # if image:
            #     enhancer_colors = ImageEnhance.Contrast(image)
            #     image = enhancer_colors.enhance(0.9)
            #print("color 6")
            #print("color 6")
            image.save("original_img."+format_1)
            img_s = cv2.imread("original_img."+format_1)
            img_hsv = cv2.cvtColor(img_s, cv2.COLOR_BGR2HSV)
            saturation = img_hsv[:, :, 1].mean()
            print("after apply 150= ",saturation)


        #buffer = BytesIO()
        #image.save(buffer, format=format_, quality=100)
        #buffer.seek(0)

        #return StreamingResponse(buffer, media_type=get_content_type(format_))
        original_image = image
        #logger.info("Enhancement Successful")
        return original_image

    except Exception as e:
        print(e)
        #logger.info("Enhancement Unsuccessful")
        pass


# @app.get("/enhancement")
# async def enhancement(Enhance_image: str):
#     """ 
#     #### The endpoint takes image url as inputs in the form of JSON, enhance the image and then return the enhanced image.\n
#     1. Enhance_image: Url of the image.
#     """

#     #print(Enhance_image)
#     response = requests.get(Enhance_image)
#     image_bytes = io.BytesIO(response.content)
#     #print(image_bytes)
#     image = PIL.Image.open(image_bytes)
#     #print(image)
#     filename = Enhance_image
#     #print(filename)
#     #this function get the format type of input image
#     def get_format(filename):
#         format_ = filename.split(".")[-1]
#         if format_.lower() == "jpg":
#             format_ = "jpeg"
#         elif format_.lower == "webp":
#             format_ = "WebP"
    
#         return format_
 
   
#     #this function for gave the same type of format to output
#     def get_content_type(format_):
#         type_ = "image/jpeg"
#         if format_ == "gif":
#             type_ = "image/gif"
#         elif format_ == "webp":
#             type_ = "image/webp"
#         elif format_ == "png":
#             type_ = "image/png"
#         #print(type_)
#         return type_

#     format_ = get_format(filename)#here format_ store the type of image by filename
    
    
#     #This function calculate the brightness of input image 
#     def calculate_brightness(image):
#         greyscale_image = image.convert('L')
#         histogram = greyscale_image.histogram()
#         pixels = sum(histogram)
#         brightness = scale = len(histogram)

#         for index in range(0, scale):
#             ratio = histogram[index] / pixels
#             brightness += ratio * (-scale + index)

#         return 1 if brightness == 255 else brightness / scale
    
#     print(calculate_brightness(image))#here print the float(calculate brightnes nummber)
    
#     image.save("original_img."+format_)
#     img_s = cv2.imread("original_img."+format_)
#     img_hsv = cv2.cvtColor(img_s, cv2.COLOR_BGR2HSV)
#     saturation = img_hsv[:, :, 1].mean()
#     print("original saturation",saturation)
                  
#     #______Here apply the Brightness and Color on image automatically according to there condition_____
#     # if (calculate_brightness(image) > 0.6 and calculate_brightness(image) < 0.7 ): 
        
#     #     enhancer_bright = ImageEnhance.Brightness(image)
#     #     image = enhancer_bright.enhance(1.2)
#     #     #print("bright 6")
#     #     if image:
#     #         enhancer_colors = ImageEnhance.Color(image)
#     #         image = enhancer_colors.enhance(1.4)
#     #         #print("color 6")
     

#     if (calculate_brightness(image) > 0.5 and calculate_brightness(image) < 0.55): 
            
#         enhancer_bright = ImageEnhance.Brightness(image)
#         image = enhancer_bright.enhance(1.2)
#         print("bright 5")
#         if image:
#             enhancer_colors = ImageEnhance.Color(image)
#             image = enhancer_colors.enhance(1.4)
#             #print("color 5")



#     if (calculate_brightness(image)  > 0.4 and calculate_brightness(image) < 0.5 ):
        
#         enhancer_bright = ImageEnhance.Brightness(image)
#         image = enhancer_bright.enhance(1.2)
#         print("bright 4")
#         if image:
#             enhancer_colors = ImageEnhance.Color(image)
#             image = enhancer_colors.enhance(1.4)
#             #print("color 4")


#     if (calculate_brightness(image)  > 0.3 and calculate_brightness(image) < 0.4):
            
#         enhancer_bright = ImageEnhance.Brightness(image)
#         image = enhancer_bright.enhance(1.5)
#         print("bright 3")
#         if image:
#             enhancer_colors = ImageEnhance.Color(image)
#             image = enhancer_colors.enhance(1.3)
#             #print("color 3")


#     if (calculate_brightness(image)  > 0.2 and calculate_brightness(image) < 0.3 ):
        
#         enhancer_bright = ImageEnhance.Brightness(image)
#         image = enhancer_bright.enhance(1.8)
#         print("bright 2")
#         if image:
#             enhancer_colors = ImageEnhance.Color(image)
#             image = enhancer_colors.enhance(1.5)
#             #print("color 2")


    
#     def calculate_brightness(image):
#         greyscale_image = image.convert('L')
#         histogram = greyscale_image.histogram()
#         pixels = sum(histogram)
#         brightness = scale = len(histogram)

#         for index in range(0, scale):
#             ratio = histogram[index] / pixels
#             brightness += ratio * (-scale + index)

#         return 1 if brightness == 255 else brightness / scale
    
#     print("after enhn b=",calculate_brightness(image))
    
#     if saturation >100 and saturation < 120 :
#         enhancer_colors = ImageEnhance.Color(image)
#         image = enhancer_colors.enhance(0.8)
#         # if image:
#         #     enhancer_colors = ImageEnhance.Contrast(image)
#         #     image = enhancer_colors.enhance(0.9)
#         #print("color 6")
#         #print("color 6")
#         image.save("original_img."+format_)
#         img_s = cv2.imread("original_img."+format_)
#         img_hsv = cv2.cvtColor(img_s, cv2.COLOR_BGR2HSV)
#         saturation = img_hsv[:, :, 1].mean()
#         print("after apply 100 =",saturation)
    
#     if saturation >120 and saturation < 150 :
#         enhancer_colors = ImageEnhance.Color(image)
#         image = enhancer_colors.enhance(0.75)
#         # if image:
#         #     enhancer_colors = ImageEnhance.Contrast(image)
#         #     image = enhancer_colors.enhance(0.9)
#         #print("color 6")
#         #print("color 6")
#         image.save("original_img."+format_)
#         img_s = cv2.imread("original_img."+format_)
#         img_hsv = cv2.cvtColor(img_s, cv2.COLOR_BGR2HSV)
#         saturation = img_hsv[:, :, 1].mean()
#         print("after apply 120=",saturation)


#     if saturation >150 :
#         enhancer_colors = ImageEnhance.Color(image)
#         image = enhancer_colors.enhance(0.6)
#         # if image:
#         #     enhancer_colors = ImageEnhance.Contrast(image)
#         #     image = enhancer_colors.enhance(0.9)
#         #print("color 6")
#         #print("color 6")
#         image.save("original_img."+format_)
#         img_s = cv2.imread("original_img."+format_)
#         img_hsv = cv2.cvtColor(img_s, cv2.COLOR_BGR2HSV)
#         saturation = img_hsv[:, :, 1].mean()
#         print("after apply 150= ",saturation)

#     buffer = BytesIO()
#     image.save(buffer, format=format_, quality=70)
#     image.show()
#     buffer.seek(0)

#     return StreamingResponse(buffer, media_type=get_content_type(format_))

class URL1(BaseModel):
    url_: List[str] = []


@app.post("/sqy_image")
async def image_scorer(image_url:URL1): 

    '''This function get image from your system or
       take input as original image
    '''

    URL1 = image_url.url_
    print(type(URL1))
    list_1 = URL1
    a_list = []

    for i in range (len(list_1)):
        URL1 = list_1[i]

        async with aiohttp.ClientSession() as session:
            async with session.get(URL1) as resp:
                contents = await resp.read()
    
        async with aiohttp.ClientSession() as session:
            async with session.get(URL1) as resp:
                contents = await resp.read()
 
        if contents == None:
            raise HTTPException(status_code=406, detail="No image found.")

        image = Image.open(BytesIO(contents))

        #this function get the format type of input image
        def get_format(filename):
            format_ = filename.split(".")[-1]
            if format_.lower() == "jpg":
                format_ = "jpeg"
            elif format_.lower == "webp":
                format_ = "WebP"
        
            return format_

        image = PIL.Image.open(BytesIO(contents))
        format_ =image.format #here format_ store the type of image by filename

        if(image):
            greyscale_image = image.convert('L')
            histogram = greyscale_image.histogram()
            pixels = sum(histogram)
            brightness = scale = len(histogram)

            for index in range(0, scale):
                ratio = histogram[index] / pixels
                brightness += ratio * (-scale + index)

        bright1 = brightness / scale
        # print(bright1)

        if(image): #here calculate the sharpness 
            image = Image.open(BytesIO(contents))
            image.save("original_img."+format_)
            # print("fifth")

            try:
                img = cv2.imread("original_img."+format_, cv2.IMREAD_GRAYSCALE)
                laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
                # print("sharpness try", laplacian_var)

                img_c = cv2.imread("original_img."+format_)
                Y = cv2.cvtColor(img_c, cv2.COLOR_BGR2YUV)[:,:,0]
                # compute min and max of Y
                min = np.min(Y)
                max = np.max(Y)

                # compute contrast
                contrast = (max-min)/(max+min)
                # print("try min=",min)

                img_s = cv2.imread("original_img."+format_)
                img_hsv = cv2.cvtColor(img_s, cv2.COLOR_BGR2HSV)
                saturation = img_hsv[:, :, 1].mean()
                # print("saturation try",saturation)

            except:
                img_s = cv2.imread("original_img."+format_) 
                img_hsv = cv2.cvtColor(img_s, cv2.COLOR_BGR2HSV)
                saturation = img_hsv[:, :, 1].mean()
                # print("saturation",saturation)

                img = cv2.imread("original_img."+format_, cv2.IMREAD_GRAYSCALE)
                laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
                # print("lap except", laplacian_var)
            
            if laplacian_var > 500:
                result = 10

            if laplacian_var > 290 and laplacian_var < 500:
                result = 9

            if laplacian_var > 135 and laplacian_var < 290:
                result = 8

            if laplacian_var > 105 and laplacian_var < 135:
                result = 7

            if laplacian_var > 80 and laplacian_var < 105:
                result = 6

            if laplacian_var > 70  and laplacian_var < 80:
                result = 5

            if laplacian_var > 60 and laplacian_var < 70:
                result = 4

            if laplacian_var > 50 and laplacian_var < 60:
                result = 3

            if laplacian_var > 45 and laplacian_var < 50:
                result = 2

            if laplacian_var > 1 and  laplacian_var < 45:
                result = 1  

            if min < 9 and laplacian_var < 100 and laplacian_var >40:
                result = 8

            if min >3 and laplacian_var >250:
                result = 8

            if saturation > 115 and laplacian_var >400:
                result = 9

            if saturation > 130 and laplacian_var >300:
                result = 8

            if saturation < 85 and laplacian_var < 50:
                result = 3

            if saturation < 103 and saturation > 85 and laplacian_var < 60 and laplacian_var >40:
                result = 6

            if min < 5 and laplacian_var < 30:
                result = 4

            if saturation > 147 and saturation <165:
                result = 7

            if saturation > 175:
                result = 3

            if saturation < 85 and laplacian_var < 50 and min > 1:
                result=3
                
            if min < 1  and laplacian_var < 40:
                result = 5

            if bright1 > 0.63 and laplacian_var < 40 and saturation < 40:
                result = 7

            if laplacian_var < 50 and min < 1 and saturation <50:
                result = 7

            if bright1 < 0.3:
                result = 3
            
            integers_to_append = result
            a_list.append(integers_to_append)
                    
            # print("result",a_list)
        
    sum1 = 0
    a_list.sort(reverse = True)
    print(a_list)

    if (len(a_list) > 10):
        print("greater than 10")

        for i in range (10):
            sum1 += a_list[i]
            # print(a_list[i])

    if (len(a_list) < 10):
        # print("less than 10")

        for i in range (len(a_list)):
            sum1 += a_list[i]
            
    result_check1 = sum1
    total_url = (len(list_1))
    result1 = result_check1/total_url
    result = (round(result1)/10)
    
    # result = str(result)
    # s1 = slice(3)
    # print(result[s1])
    print("Score = ",result)
    
    return ({"score_":result})


@app.post("/enhancement_logo_without_ext")
async def enhancement_logo_without_ext(image_details: ImageDetails):
    """ 
    #### The endpoint takes image url without extension as inputs in the form of JSON ,enhance the image and then pastes the Square Yards logo as a watermark on the input images and then compresses it and returns file without extension.\n
    1. url_: Url of the image.
    2. width_percentage: Size of watermark based on the width of the image. Range (0-1).
    3. compression_info: Details regarding image compression.
    4. position: position of logo on image.
    """
    URL1 = image_details.url_
    URL = image_details.url_
    #logger.info(URL)
    #total_req_ext(1)
    ##logger.info("Total number of request without ext: {}".format(total_request_extension.count(1)))
    response = requests.get(URL)
    #print(response)
    img = Image.open(BytesIO(response.content))
    #print(img)
    #print(img.format.lower())

    URL = URL + "." + img.format.lower()
    
    width_percentage = image_details.width_percentage

    position = image_details.position

    filename, original_image = await get_image_properties(URL, width_percentage, position)
    original_image = await get_body(image_details.url_)
    

    try:

        squareyard_logo = SQUARE_YARDS_LOGO.copy()
        original_image, format_, quality = get_final_image(image_details, original_image, width_percentage, squareyard_logo, position, filename)
        buf = BytesIO()
        if format_ == 'gif':
            frames = [get_final_image(image_details, frame.copy(), width_percentage, squareyard_logo, position, filename)[0] for frame in ImageSequence.Iterator(original_image)]
            frames[0].save(buf, save_all=True, append_images=frames[1:], format=format_, quality=quality, optimize=True)
        elif format_ == 'png':
            format_ = 'webp'
            original_image.save(buf, format=format_, quality = 90, optimize=True)
        else:
            original_image.save(buf, format=format_, quality = 90, optimize=True)

            
    except Exception as e:
            print(e)
            #logger.info("Error: HTTPException(status_code=500, detail= Error while processing the image.")
            raise HTTPException(status_code=500, detail="Error while processing the image.")
    buf.seek(0)

    filename = filename.replace(filename.split(".")[-1], '')
    


    filename = filename.replace(".","")
    
    #print(filename)
    
    #logger.info("Result: Successful")
    #s1=sample_list_ext(1)
    ##logger.info("Successful Response without ext: {}".format(sample_list_for_without_exten.count(1)))
    return StreamingResponse(buf, media_type=get_content_type(format_), headers={'Content-Disposition': 'inline; filename="%s"' %(filename,)})

@app.post("/sqy_extension")
async def extension(check_image: URL): 

    '''This function get image URL from user and tell 
    the image extension as a output
    '''
    URL = check_image.url_
    response = requests.get(URL)

    image_bytes = io.BytesIO(response.content)
    image = PIL.Image.open(image_bytes) 

    try:
        result1 = "."+image.format.lower()
        URL11 = URL + result1

    except Exception:
        raise HTTPException(status_code=406, detail="Not a valid URL")


    if URL11.lower().endswith((".jpg", ".png", ".jpeg", ".gif", ".webp",".jfif")) == False:
        raise HTTPException(status_code=406, detail="Not a valid URL")


    output1 = result1   
    return ({"ext_":output1})

@app.post("/enhancement_logo_without_ext2")
async def enhancement_logo_without_ext(image_details: ImageDetails):
    """ 
    #### The endpoint takes image url without extension as inputs in the form of JSON ,enhance the image and then pastes the Square Yards logo as a watermark on the input images and then compresses it, return file with extension.\n
    1. url_: Url of the image.
    2. width_percentage: Size of watermark based on the width of the image. Range (0-1).
    3. compression_info: Details regarding image compression.
    4. position: position of logo on image.
    """
    URL1 = image_details.url_
    URL = image_details.url_
    #logger.info(URL)
    #total_req_ext2(1)
    ##logger.info("Total number of request without ext2: {}".format(total_request_extension2.count(1)))
    response = requests.get(URL)
    #print(response)
    img = Image.open(BytesIO(response.content))

    print(img)
    #print(img.format.lower())

    URL = URL + "." + img.format.lower()
    
    width_percentage = image_details.width_percentage

    position = image_details.position

    filename, original_image = await get_image_properties(URL, width_percentage, position)
    original_image = await get_body(image_details.url_)
    

    try:

        squareyard_logo = SQUARE_YARDS_LOGO.copy()
        original_image, format_, quality = get_final_image(image_details, original_image, width_percentage, squareyard_logo, position, filename)
        buf = BytesIO()
        if format_ == 'gif':
            frames = [get_final_image(image_details, frame.copy(), width_percentage, squareyard_logo, position, filename)[0] for frame in ImageSequence.Iterator(original_image)]
            frames[0].save(buf, save_all=True, append_images=frames[1:], format=format_, quality=quality, optimize=True)
        elif format_ == 'png':
            format_ = 'webp'
            original_image.save(buf, format=format_, quality = 90, optimize=True)
        else:
            original_image.save(buf, format=format_, quality = 90, optimize=True)

            
    except Exception as e:
            print(e)
            #logger.info("Error: HTTPException(status_code=500, detail= Error while processing the image.")
            raise HTTPException(status_code=500, detail="Error while processing the image.")
    buf.seek(0)
    
    #logger.info("Result: Successful")
    #s1=sample_list_ext2(1)
    ##logger.info("Successful Response without ext2: {}".format(sample_list_for_without_exten2.count(1)))
    return StreamingResponse(buf, media_type=get_content_type(format_), headers={'Content-Disposition': 'inline; filename="%s"' %(filename,)})

@app.post("/enhancement_logo")
async def enhancement_logo(image_details: ImageDetails):
    """ 
    #### The endpoint takes multiple parameters as inputs in the form of JSON ,enhance the image and then pastes the Square Yards logo as a watermark on the input images and then compresses it.\n
    1. url_: Url of the image.
    2. width_percentage: Size of watermark based on the width of the image. Range (0-1).
    3. compression_info: Details regarding image compression.
    4. position: position of logo on image.
    """
    URL = image_details.url_
    #logger.info(URL)
    total_req_logo_enhancement(1)
    #logger.info("Total number of request logo enhancement: {}".format(total_request_logo_enhancement.count(1)))
    width_percentage = image_details.width_percentage

    position = image_details.position
    
    filename, original_image = await get_image_properties(URL, width_percentage, position)

    original_image = await get_body(image_details.url_)
    

    try:

        squareyard_logo = SQUARE_YARDS_LOGO.copy()
        original_image, format_, quality = get_final_image(image_details, original_image, width_percentage, squareyard_logo, position, filename)
        buf = BytesIO()
        if format_ == 'gif':
            frames = [get_final_image(image_details, frame.copy(), width_percentage, squareyard_logo, position, filename)[0] for frame in ImageSequence.Iterator(original_image)]
            frames[0].save(buf, save_all=True, append_images=frames[1:], format=format_, quality=quality, optimize=True)
        elif format_ == 'png':
            format_ = 'webp'
            original_image.save(buf, format=format_, quality = 70, optimize=True)
        else:
            original_image.save(buf, format=format_, quality = 70, optimize=True)

            
    except Exception as e:
            print(e)
            #logger.info("Error: HTTPException(status_code=500, detail= Error while processing the image.)")
            raise HTTPException(status_code=500, detail="Error while processing the image.")
    buf.seek(0)

    #logger.info("Result: Successful")
    sample_list_logo_enhancement(1)
    #logger.info("Successful Response logo enhancement: {}".format(sample_list_for_logo_enhancement.count(1)))
    return StreamingResponse(buf, media_type=get_content_type(format_), headers={'Content-Disposition': 'inline; filename="%s"' %(filename,)})

@app.post("/addWatermark")
async def add_watermark(image_details: ImageDetails):
    """ 
    #### The endpoint takes multiple parameters as inputs in the form of JSON, pastes the Square Yards logo as a watermark on the input images and then compresses it.\n
    1. url_: Url of the image.
    2. width_percentage: Size of watermark based on the width of the image. Range (0-1).
    3. compression_info: Details regarding image compression.
    4. position: position of logo on image.
    """
    URL = image_details.url_
    width_percentage = image_details.width_percentage

    position = image_details.position
    
    filename, original_image = await get_image_properties(URL, width_percentage, position)

    try:
        squareyard_logo = SQUARE_YARDS_LOGO.copy()
        original_image, format_, quality = get_final_image(image_details, original_image, width_percentage, squareyard_logo, position, filename)
        buf = BytesIO()
        if format_ == 'gif':
            frames = [get_final_image(image_details, frame.copy(), width_percentage, squareyard_logo, position, filename)[0] for frame in ImageSequence.Iterator(original_image)]
            frames[0].save(buf, save_all=True, append_images=frames[1:], format=format_, quality=quality, optimize=True)
        elif format_ == 'png':
            format_ = 'webp'
            original_image.save(buf, format=format_, quality = 70, optimize=True)
        else:
            original_image.save(buf, format=format_, quality = 70, optimize=True)

            
    except Exception as e:
            print(e)
            #logger.info("Error: HTTPException(status_code=500, detail=Error while processing the image.")
            raise HTTPException(status_code=500, detail="Error while processing the image.")
    buf.seek(0)

    #logger.info("Result: Successful")
    return StreamingResponse(buf, media_type=get_content_type(format_), headers={'Content-Disposition': 'inline; filename="%s"' %(filename,)})
 
 
@app.post("/addWatermarkIC")
async def add_watermarkIC(image_details: ImageDetails):
    """ 
    #### The endpoint takes multiple parameters as inputs in the form of JSON and pastes the Interior Company logo as a watermark on the input images.\n
    1. url_: Url of the image.
    2. width_percentage: Size of watermark based on the width of the image. Range (0-1).
    3. compression_info: Details regarding image compression.
    4. position: position of logo on image.
    """
    URL = image_details.url_
    width_percentage = image_details.width_percentage
    position = image_details.position
    filename, original_image = await get_image_properties(URL, width_percentage, position)
    
    try:
        ic_logo = IC_LOGO.copy()
        original_image, format_, quality = get_final_image(image_details, original_image, width_percentage, ic_logo, position, filename)
        buf = BytesIO()
        if format_ == 'gif':
            frames = [get_final_image(image_details, frame.copy(), width_percentage, ic_logo, position, filename)[0]\
                         for frame in ImageSequence.Iterator(original_image)]
            frames[0].save(buf, save_all=True, append_images=frames[1:], format=format_, quality=quality, optimize=True)
        else:
            original_image.save(buf, format=format_, quality=quality, optimize=True)
    except Exception as e:
            print(e)
            #logger.info("Error: HTTPException(status_code=500, detail= Error while processing the image.")
            raise HTTPException(status_code=500, detail="Error while processing the image.")
    buf.seek(0)
    #logger.info("Result1: Successful")
    return StreamingResponse(buf, media_type=get_content_type(format_), headers={'Content-Disposition': 'inline; filename="%s"' %(filename,)})


@app.get('/error_logs')
async def get_gunicorn_error_logs():
    path = os.path.join(os.getcwd(), 'gunicorn-error.log')
    log_path = os.environ.get("ERROR_LOGFILE", path)
    data = ""
    try:
        with open(log_path, 'r') as f:
            data += "<ul>"
            for s in f.readlines():
                data += "<li>" + str(s) + "</li>"
            data += "</ul>"
    except:
        pass
    return HTMLResponse (content=data)
