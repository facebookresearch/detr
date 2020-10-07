from pycocotools.coco import COCO
import pandas as pd
import json 
import numpy as np 
import os
from json import JSONEncoder
from PIL import Image
import datetime
import json

# class to convert numpy arrays to json 
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
        
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class cocoConvert:
    """ cocoConvert a module that could be used to convert between coco, VIA and LabelBox. 
        It is meant for easier analyis of different types format of annotations data
        
        Currently supported Conversions:
            coco        ->  VIA, LabelBox
            VIA         ->  coco, VIAJSON
            LabelBox    ->  VIA, COCO, VIAJSON
    """
    
    def __init__(self, inputFile, outputFile, imagesPath, fromType ):
        """
        Parameters
        ----------
        inputFile : path or dataFrame
            The file to be converted
            
        outputFile : the output file name and path 
        
        imagesPath : the path of the location of the iamges
        
        fromType : str
            Supported types "coco", "VIAJSON", "LabelBoxCSV", "VIA"
            
        """
        
    
        self.fromType = fromType
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.imagesPath = imagesPath
        
        if isinstance(self.inputFile, str) and os.path.exists(self.inputFile):
        
            if fromType == "coco":
                self.inputData =  COCO(self.inputFile)
            elif fromType != "VIAJSON":
                self.inputData = pd.read_csv(inputFile)
            elif fromType == "VIAJSON":
                self.inputData = json.load(open(inputFile))
        else:
            self.inputData = inputFile

    def convertToVia(self):
        """ convert to via from labelbox or coco"""
    
        annotations = []
        if self.fromType == "coco":
            data = self.__convertCOCO_VIA()
            
        elif self.fromType == "LabelBoxCSV":
            data = self.__convertLB_VIA()
        
        else:
            raise Exception("File type "+ self.fromType + " is not supported")
        
        return pd.DataFrame(data= data,columns = ['filename', 'file_size', 'file_attributes', 'region_count', 'region_id',
       'region_shape_attributes', 'region_attributes'] )             
    
    def convertToViaJson(self):
        """convert to VIAJSON -> this could be dfrom VIA. LabelboxCSV or coco"""
        
        annotations = []
        if self.fromType == "VIA":
            data = self.__convertVIA_VIAJSON()
            
        elif self.fromType == "LabelBoxCSV":
            tmp = self.inputData
            self.inputData = self.__convertLB_VIA()
            data = self.__convertVIA_VIAJSON()
            self.inputData = tmp
            
        elif self.fromType == "coco":
       
            tmp = self.inputData
            self.inputData  = self.convertToVia()          
            data = self.__convertVIA_VIAJSON()
            self.inputData = tmp
            
        else:
            raise Exception("File type "+ self.fromType + " is not supported")
        
        return data
    
    def convertToCOCO(self, categories=None,super_categories=['N/A'],first_class_index=1, output_file_name = None):
        """convert to coco """
    
        if output_file_name is None:
           output_file_name = self.outputFile
    
        if self.fromType == "VIAJSON":
        
           data = self.__convertVIA_COCO(   annpath = self.inputData, \
                                            imgdir = self.imagesPath,\
                                            categories=categories,\
                                            super_categories=super_categories,\
                                            output_file_name=output_file_name,\
                                            first_class_index=first_class_index)
            
                                            
        elif self.fromType == "VIA":
            tmp = self.inputData
            self.inputData = self.__convertVIA_VIAJSON()
            data = self.__convertVIA_COCO(  annpath = json.loads(self.inputData), \
                                            imgdir = self.imagesPath,\
                                            categories=categories,\
                                            super_categories=super_categories,\
                                            output_file_name=output_file_name,\
                                            first_class_index=first_class_index)
            self.inputData = tmp
            
        else:
            raise Exception("File type "+ self.fromType + " is not supported")
            
        
            
        return data
 
    def __convertVIA_VIAJSON(self):
    
        #VIAJSON format for each image: <fileName>: {<filename>, <size>, <regions>{<shape _attributes>, <region_attributes>}, <file_attributes>}
        df = self.inputData.copy()

        df.reset_index(drop=True, inplace = True)
        mergedBBoxLocAndNames = []
         
        #contruct <regions> by merging shape and region attributes 
        for i in range(df.shape[0]):           
            mergedBBoxLocAndNames.append( {'shape_attributes': json.loads(df.loc[i, 'region_shape_attributes']),\
                                           "region_attributes":json.loads(df.loc[i,'region_attributes'])})
        
         ##aggregate all bboxes and classes for unique filenames
        df['bbox_shapes_names'] = mergedBBoxLocAndNames
        dfgrouped = pd.DataFrame( data = df.groupby(['filename', 'file_size'])['bbox_shapes_names'].apply(lambda x : x.tolist()))
        
        #reconstruct into json format
        jsonConv = {}
        for i in dfgrouped.reset_index().values:
            jsonConv[i[0]] = {'filename':i[0], "size":i[1], 'regions': i[2], 'file_attributes': {}}
            
        if self.outputFile:
            with open(self.outputFile, 'w') as outfile:
                json.dump(jsonConv, outfile, cls=NumpyArrayEncoder)
                
        return json.dumps(jsonConv, cls=NumpyArrayEncoder)
     

     
    def __convertCOCO_VIA(self):
    
        #get the categories annotations and images from coco file
        cats = self.inputData.loadCats(self.inputData.getCatIds())
        images = self.inputData.loadImgs(self.inputData.getImgIds())
        annotations = self.inputData.loadAnns(self.inputData.getAnnIds())
        
        finalDataFrame = []
        for annot in annotations:
            
            imageIndex = annot['image_id']
            
            imageName = self.getIndex(imageIndex, images)['file_name']

            try:
                finalDataFrame.append(
                    [
                        #filename
                        imageName,
                        #filesze
                        os.path.getsize(os.path.join(self.imagesPath, imageName)),
                        #file attributes
                        {},
                        #region count
                        0,
                        #region id
                        0,  
                        #bbox in json format. only name:rect supported                         
                        json.dumps({"name":"rect", "x":annot['bbox'][0], "y":annot['bbox'][1], \
                        "width":annot['bbox'][2], "height":annot['bbox'][3]}),
                        
                        #the class name. should always be named class
                        json.dumps({"class":self.getIndex(annot['category_id'], cats)["name"]})       
                    ]
                )
            except:
                print('probably missing image or annotation Filename: ' , imageName)
                
        return finalDataFrame
    
    def __convertLB_VIA(self):
        finalDf = []
        for i in self.inputData.values:
            objects = {}
            imageName= i[9]
            try:
                objects = json.loads(i[3])['objects']
            except (ValueError, IndexError, KeyError):
                print("\n BBInfo missing at:")
                continue
            
            region_index = 0
            for object1 in objects:
                  
                obj = [                 
                    imageName, 
                    os.path.getsize(os.path.join(self.imagesPath, imageName)) if self.imagesPath else ""
                    
                    ,
                    {},
                    len(objects),
                    region_index,
                    
                    json.dumps({"name":"rect", "x":object1['bbox']['left'], "y":object1['bbox']['top'],\
                    "width":object1['bbox']['width'], "height":object1['bbox']['height']}),
                    
                    json.dumps({"class":object1['value']})
                     
                ]
               
                region_index +=1
                
                finalDf.append(obj)
        return finalDf
        
    def __convertVIA_COCO(self,
        annpath, 
        imgdir,
        categories=None,
        super_categories=None,
        output_file_name = None,
        first_class_index=1,
    ):
        
        if categories is None:
            print("please enter categories") 
            

        coco = { 'info':{}, 'images':[], 'annotations':[], 'licenses':[], 'categories':[] } 
        
        default_category = categories[0]

        category_dict = dict()
        for (cat_id, cat_name) in enumerate(categories, start=first_class_index):
            category_dict[cat_name] = cat_id

        if super_categories is None:
            default_super_category = "bone"
            super_categories = [default_super_category for _ in categories]

        # compute the info and lisence
        coco["info"] = {
            "description": "Fridge",
            "url": "",
            "version": "0.1.0",
            "year": 2020,
            "contributor": "lyupo",
            "date_created": datetime.datetime.utcnow().isoformat(" "),
        }
        coco["licenses"] = [
            {
                "id": 1,
                "name": "Lyupo",
                "url": "",
            }
        ]
        
        # get the categories. Does not support suppercategories
        coco["categories"] = [
            {
                "id": category_dict[cat_name],
                "name": cat_name,
                "supercategory": super_categories[0],
            }
            for cat_name in categories
        ]
     
     
        ann=  annpath
      
        ann_id = 0
        
        #itterate over the annotations
        for img_id, key in enumerate(ann.keys()):
            try:
                filename = ann[key]["filename"]
            except KeyError:
                print("Plase make sure that the input data is in the format of VIA JSON")
                
            img = Image.open(os.path.join(imgdir , filename))
            image_size = img.size
            
            coco["images"].append( {
              'id':         img_id,
              'width':      image_size[0],
              'height':     image_size[1],
              'file_name':  os.path.basename(filename),
              'license':    0,
              'flickr_url':"",
              'coco_url':"",
              'date_captured':'',
                })
            
            #regions in VIAJson represent the actual bboxes and their corresponding classes
            regions = ann[key]["regions"]

            # for one image ,there are many regions,they share the same img id
            for region in range(len(regions)):
                #get bbox info
                region_attributes = regions[region]["region_attributes"]
                
                #get the class from VIAJSON
                try:
                   cat_name = region_attributes["class"]           
                   cat_id = category_dict[cat_name]                
                except KeyError:
                    print("Skipping unknown category {} in {}".format(cat_name, filename))
                    continue
                
                shape_attributes = regions[region]["shape_attributes"]
                
                #get the bbox in the format of coco -> x0, y0, w, h, area, segmentation..
                annotation  = self.shape_to_coco_annotation(shape_attributes)
                
                cat_id = category_dict[cat_name]

                ann_info = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "iscrowd": annotation['iscrowd'],
                    "area": annotation['area'],  # float
                    "bbox": annotation['bbox'],  # [x,y,width,height]
                    "segmentation": annotation['segmentation'], 
                }

                coco["annotations"].append(ann_info)
                ann_id = ann_id + 1

        if output_file_name is not None:
            print("Saving to {}".format(output_file_name))

            with open(output_file_name, "w") as f:
                json.dump(coco, f)

        return coco

    def shape_to_coco_annotation(self, shape_attributes):
        """Transforms bbox into coco-style bbox info. Currently supports only bounding boxes"""
        
        annotation = { 'segmentation':[[]], 'area':[], 'bbox':[], 'iscrowd':0 }
        
        if shape_attributes['name'] == 'rect':
                x0 = shape_attributes['x']
                y0 = shape_attributes['y']
                w  = shape_attributes['width']
                h  = shape_attributes['height']
                x1 = x0 + w;
                y1 = y0 + h;
                annotation['segmentation'][0] = [x0, y0, x1, y0, x1, y1, x0, y1];
                annotation['area'] =  w * h ;
                annotation['bbox'] = [x0, y0, w, h]
                
        else: 
            raise ValueError("Only annotation of type rect is available")
                         
        return annotation
    
        
    def getIndex(self, idx, obj):
        ## get a specific index from coco formated json
        for obj in obj:
            if(obj['id']==idx):
                return obj
                
                
  
    