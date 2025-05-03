import os
import numpy as np
import pandas as pd
import numpy as np
import skimage.io as io

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg 
from detectron2.utils.visualizer import Visualizer

from PIL import Image

def pred_price(image_path):

    def DICE_COE(mask1, mask2):
        intersect = np.sum(mask1*mask2)
        fsum = np.sum(mask1)
        ssum = np.sum(mask2)
        dice = (2 * intersect ) / (fsum + ssum)
        #dice = round(dice, 3) # for easy reading
        return dice  

    #damage_type model

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this  dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (damage)
    cfg.MODEL.RETINANET.NUM_CLASSES = 3 # only has one class (damage)

    cfg.MODEL.DEVICE = "cpu"

    cfg.MODEL.WEIGHTS = os.path.join('models','model2',"model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.57  # set a custom testing threshold for this model
    predictor1 = DefaultPredictor(cfg)

    metadata = {'thing_classes':['minor', 'moderate', 'severe']}
    im = io.imread(image_path)
    outputs = predictor1(im)
   
    v = Visualizer(im[:, :, ::-1],
                    metadata=metadata,
                    scale=0.5, 
                    #instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    ima = Image.fromarray(out.get_image()[:, :, ::-1])
    
    save_path = 'pred_'+image_path.split('\\')[1]
    ima.save('static/'+save_path)

    #car_part model
    cfg_2 = get_cfg()
    cfg_2.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg_2.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg_2.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this  dataset (default: 512)
    cfg_2.MODEL.ROI_HEADS.NUM_CLASSES = 5  # only has one class (damage)
    cfg_2.MODEL.RETINANET.NUM_CLASSES = 5 # only has one class (damage)

    cfg_2.MODEL.DEVICE = "cpu"

    cfg_2.MODEL.WEIGHTS = os.path.join('models','model3',"model_final.pth")
    cfg_2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.70  # set a custom testing threshold for this model
    predictor2 = DefaultPredictor(cfg_2)

    cost_matrix = pd.DataFrame({
    'headlamp': {'minor': 1000, 'moderate': 3000, 'severe': 6000},
    'rear_bumper': {'minor': 1500, 'moderate': 4000, 'severe': 8000},
    'door': {'minor': 2000, 'moderate': 5000, 'severe': 9000},
    'hood': {'minor': 1800, 'moderate': 4500, 'severe': 8500},
    'front_bumper': {'minor': 1500, 'moderate': 4000, 'severe': 8000}
})
    #creating damage features

    dict = {'headlamp_dice':[],'rear_bumper_dice':[],'door_dice':[],'hood_dice':[],'front_bumper_dice':[],'minor':[],'moderate':[],'severe':[]}

    damage_categories = ['minor','moderate','severe']
    part_categories = ['headlamp','rear_bumper','door','hood','front_bumper']
    img = io.imread(image_path)
    damage_type_outputs = outputs
    car_part_outputs = predictor2(img)

    cat_ids1 = [0,1,2]
    anns = damage_type_outputs['instances'].pred_masks.cpu().numpy()

    cat_ids2 = [0,1,2,3,4]
    anns2 = car_part_outputs['instances'].pred_masks.cpu().numpy()

    global headlamp_dice
    global rear_bumper_dice
    global door_dice
    global hood_dice
    global front_bumper_dice

    for j in range(len(anns)):


        headlamp_dice = 0
        rear_bumper_dice = 0
        door_dice = 0
        hood_dice = 0
        front_bumper_dice = 0
        cats = []
        
        mask1 = anns[j]

        for k in range(len(anns2)):
            mask2 = anns2[k]
            dice_coe = DICE_COE(mask1,mask2)
            part_category_id = int(car_part_outputs['instances'].pred_classes[k])
            part_affected = part_categories[part_category_id]
            cats.append(int(car_part_outputs['instances'].pred_classes[k]))
            globals()[part_affected+'_dice'] += dice_coe
        
        for k in cat_ids2:
            part_name = part_categories[k]
            dict[part_name+'_dice'].append(globals()[part_name+'_dice'])

        damage_type = damage_categories[int(damage_type_outputs['instances'].pred_classes[j])]
        dict[damage_type].append(1)

        for k in cat_ids1:
            if k != int(damage_type_outputs['instances'].pred_classes[j]):
                damage_type = damage_categories[k]
                dict[damage_type].append(0)
    
    val_repair_cost_dataset = pd.DataFrame(dict)

    unknown = []
    for i in val_repair_cost_dataset.iloc:
        if i['headlamp_dice'] == i['rear_bumper_dice'] == i['door_dice'] == i['hood_dice'] == i['front_bumper_dice'] == 0:
            unknown.append(1)
        else:
            unknown.append(0)

    val_repair_cost_dataset.insert(loc=5, column='unknown', value=unknown)

    #calculate total_price
    def calculate_price(val_repair_cost_dataset, cost_matrix):
        total_price = 0

        # Iterate through each row in the dataset
        for _, row in val_repair_cost_dataset.iterrows():
            part_cost = 0

            # Calculate cost for each part based on the overlap percentage and damage type
            for part in ['headlamp', 'rear_bumper', 'door', 'hood', 'front_bumper']:
                overlap = row[f"{part}_dice"]  # Get the Dice Coefficient
                if overlap == 0.0:
                    continue
                # Check the damage type and add the corresponding cost
                if row['minor'] == 1:
                    part_cost += overlap * cost_matrix[part]['minor']
                elif row['moderate'] == 1:
                    part_cost += overlap * cost_matrix[part]['moderate']
                elif row['severe'] == 1:
                    part_cost += overlap * cost_matrix[part]['severe']

            # Add the part cost to the total price
            total_price += part_cost

        return total_price

    total_price = round(calculate_price(val_repair_cost_dataset, cost_matrix))
    return total_price, save_path


if  __name__ == '__main__':
    price,path = pred_price('static/11.jpg')
    print(price)