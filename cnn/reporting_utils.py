import pandas as pd 
import numpy as np 
import json 
import matplotlib.pyplot as plt
from image_utils import load_image
from model_util import predict
from data_loader_util import evaluate_dir

def get_expected_labels(predictions,classes):
    labels = []
    for p in predictions:
        labels.append(classes[p])
    return  labels

def create_report(path,expected,model,shape,classes,grey_scale=False):
    predictions  = evaluate_dir(path,model,shape,classes,grey_scale)
    expected = expected['predictions'].values
    expected_labels = get_expected_labels(expected,classes)
    df = pd.DataFrame({
        'file':predictions[0] ,
        'predictions':predictions[1],
        'predicted_labels':predictions[2],
        'expected':expected,
        'expected_labels':expected_labels
        })

    total_number_of_files  = len(predictions[0])
    all_data = df.to_json(orient="split")
    
    all_df = df 

    comparison_column = np.where(all_df["predictions"] == all_df["expected"], True, False)
    all_df["equal"] = comparison_column

    df = all_df[all_df['equal'] == False]
    wrongly_identified_data = df.to_json(orient="split")
    no_of_wrongly_identified_files = len(df["file"].values)

    df = all_df[all_df['equal'] == True]
    correctly_identified_data = df.to_json(orient="split")

    percentage_error = 100 - (( total_number_of_files - no_of_wrongly_identified_files ) / total_number_of_files ) * 100
    return (all_data,wrongly_identified_data,correctly_identified_data,percentage_error)


def append_all_rows_to_result(all_rows,result):
    data = json.loads(all_rows)['data']
    columns = json.loads(all_rows)['columns']
    rows = [] 
    for row in data:
        row_map = {}
        for index,item in enumerate(row) :
            row_map[columns[index]] = item 
        rows.append(row_map)  
    result['all'] = rows
    return result 

def append_rows_with_errors_result(wrongly_identified,result):
    data = json.loads(wrongly_identified)['data']
    columns = json.loads(wrongly_identified)['columns']
    rows = [] 
    for row in data:
        row_map = {}
        for index,item in enumerate(row[0:len(row)-1]) :
            row_map[columns[index]] = item 
        rows.append(row_map)       
    result['errors'] = rows
    return result 

def append_rows_with_correct_result(correctly_identified_data,result):
    data = json.loads(correctly_identified_data)['data']
    columns = json.loads(correctly_identified_data)['columns']
    rows = [] 
    for row in data:
        row_map = {}
        for index,item in enumerate(row[0:len(row)-1]) :
            row_map[columns[index]] = item 
        rows.append(row_map)       
    result['correct'] = rows
    return result 


def batch_predict(path='',model=None,shape=None,classes=None,correct_predictions=None,show_all=False,show_correct=False,show_errors = True, grey_scale=False ):
    result = {}
    all_rows,wrongly_identified,correctly_identified_data,percentage_error = create_report(path, correct_predictions, model,shape,classes, grey_scale)
    if show_all: 
        result = append_all_rows_to_result(all_rows,result)

    if show_errors: 
        result = append_rows_with_errors_result(wrongly_identified,result)

    if show_correct:
        result = append_rows_with_correct_result(correctly_identified_data,result)

    result['percentage_error'] = percentage_error
    return result 

def predict_single(path='',model=None,shape=None, classes=None, show_image=False,grey_scale=False):
    img = load_image(path,(shape[0],shape[1]),grey_scale)
    if show_image:
        plt.imshow(img)
    return predict(model,img,shape,classes)

def predict_images_from_test_set(model,images,classes,index):
    plt.imshow(images[index])
    predictions = model.predict(images)
    label = np.argmax(predictions[index])
    return classes[label]