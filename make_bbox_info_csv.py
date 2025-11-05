import pandas as pd
import json 
import logging
import pydicom
import click

from damply import dirs
from pathlib import Path
from typing import Generator


def convert_pydicom_to_dict(pydcm_dataset: pydicom.dataset.Dataset):
    '''
    Recursively converts a pydicom dataset into a nested dictionary for easier search capabilities later on. 
    Function gotten from https://github.com/pydicom/pydicom/issues/319

    Parameters
    ----------
    pydcm_dataset: pydicom.dataset.Dataset 
        A structured report DICOM that has been converted into a pydicom dataset object

    Returns
    ----------
    dcm_dict: dict
        A Python dictionary with all DICOM information stored in it. Keys are the element value representations
    '''
    dcm_dict = dict() 
    for element in pydcm_dataset: # Go through all element data listed in the dataset
        if element.VR != "SQ": #If the element's value representation is not a sequence
            dcm_dict[element.name] = str(element.value) #Store the DICOM header info with keys as element names
        else: 
            #Call function recursively for each element stored in the sequence
            dcm_dict[element.name] = [convert_pydicom_to_dict(item) for item in element]
        
    return dcm_dict

def get_dcm_tag_info(dicom_dict: dict, 
                     dicom_tag: str, 
                     curr_path: list = None, 
                     dicom_value: str = None
                    ): 
    '''
    Recursively searches for a specific DICOM tag using the inputed value representation and outputs the key-value pairs that have that information. 
    Also gets the location in the dictionary at which those key-value pairs were found.  
    Can begin from the root or from a specific node of the structure. Option to only find keys with a specific value. 

    Parameters
    ----------
    dicom_dict: dict
        A dictionary with all DICOM information stored in it. Keys are element value representations. 
    dicom_tag: str
        The DICOM tag you are trying to search for. Expects capitalization and spacing.
    curr_path: list
        The path to a specific part of the nested dictionary stored in a list. To be used in recursion. 
        Leave this as None when initially calling function. Subset dictionary first if you don't want to look through all of it. 
    dicom_value: str
        Option to only return keys with a specific value inputted here.

    Returns
    ----------
    key_val_loc: list
        A list containing the searched DICOM tag, the values associated with that term, and the location at which it was found in the dictionary.
    '''

    if curr_path is None: 
        curr_path = []

    # Check if value is a list (this is for multiple nested dictionaries found at the same node)
    if isinstance(dicom_dict, list): 
        for index, value in enumerate(dicom_dict): 
            new_path = list(curr_path)
            new_path.append(index)

            for found_tag in get_dcm_tag_info(value, 
                                                dicom_tag, 
                                                curr_path = new_path, 
                                                dicom_value = dicom_value): 
                if dicom_value is not None: 
                    #Check if the current tag's value matches the one inputted 
                    if found_tag[1] == dicom_value: 
                        yield found_tag
                    else: 
                        continue
                else: 
                    #Entering here --> output all found matches for the inputted DICOM tag
                    yield found_tag

    # Check if value is a dictionary 
    if isinstance(dicom_dict, dict): 
        for key, value in dicom_dict.items():
            new_path = list(curr_path)
            new_path.append(key)
            for found_tag in get_dcm_tag_info(value, 
                                                dicom_tag, 
                                                curr_path = new_path,
                                                dicom_value = dicom_value): 
                yield found_tag

            if key == dicom_tag: 
                new_path = list(curr_path)
                new_path.append(key)
                key_val = [key, value, new_path] #Output the search term, value found in correspondance to that term, and the location it was found at

                if dicom_value is not None: 
                    #Check if the current tag's value matches the one inputted 
                    if key_val[1] == dicom_value: 
                        yield key_val
                    else: 
                        continue
                else: 
                    #Entering here --> output all found matches for the inputted DICOM tag
                    yield key_val

def dcm_list_to_df(dcm_tag_gen_funct: Generator): 
    '''
    Takes all information gathered from get_dcm_tag_info and converts it into a pandas dataframe. 

    Parameters
    ----------
    dcm_tag_gen_funct: Generator
        The iterator that is created from running get_dcm_tag_info function
    
    Returns
    ----------
    dcm_info_df: pd.DataFrame
        All information gathered from get_dcm_tag_info (search term, value, location) organized into a dataframe.
    '''
    dcm_info_cols = ['SearchTerm', 
                     'Value', 
                     'Location']
    
    dcm_info_df = pd.DataFrame(dcm_tag_gen_funct, columns = dcm_info_cols)

    return dcm_info_df

def dcm_dicts_to_df(dicom_dict: dict, 
                    dicom_tag_list: list, 
                    dicom_value_list: list): 
    '''
    From the DICOM dictionary, systematically look for tag (and value pair if applicable), find their locations and values, 
    and concatenate all into one dataframe. 

    Parameters
    ----------
    dicom_dict: dict
        A converted pydicom dataset object containing all DICOM information for a structured report
    dicom_tag_list: list
        A list of DICOM tags to search for. Expects value representation form with capitalization and spaces
    dicom_value_list: list 
        A list of values associated with the DICOM tags. Must be presented in same order as dicom_tag_list. 
        If you are not searching for a particular value in a tag, fill the index with None. Must be the same length as the tag list

    Returns
    ----------
    all_dcm_info_df: pd.DataFrame
        A dataframe containing all information searched for and its location in the dicom_dict. 
    '''
    
    for idx in range(len(dicom_tag_list)): 
        curr_tag = dicom_tag_list[idx]
        curr_value = dicom_value_list[idx]

        #Find all information corresponding to the current tag (and value if applicable)
        curr_matched_info = get_dcm_tag_info(dicom_dict = dicom_dict,
                                             dicom_tag = curr_tag, 
                                             dicom_value = curr_value)
        
        #Convert all found information into dataframe format
        curr_info_df = dcm_list_to_df(dcm_tag_gen_funct = curr_matched_info)

        #Concatenate to the total information dataframe 
        if 'all_dcm_info_df' not in locals(): #Check if the dataframe variable doesn't exist yet
            all_dcm_info_df = curr_info_df
        else: 
            all_dcm_info_df = pd.concat([all_dcm_info_df, curr_info_df], ignore_index = True).reset_index(drop = True)

    return all_dcm_info_df

def find_best_list_match(ref_loc: list, 
                         loc_lists: list): 
    '''
    Takes a list of lists with dictionary location information in them and finds the closest match(es) (preserving order) to the reference list.

    Parameters
    ----------
    ref_loc: list
        The list of locations you are going to compare all other lists to
    loc_lists: list
        A list of lists including all location information you would like to search through.

    Returns
    ----------
    related_locations: list
        A list of lists with all matched locations
    '''
    # Check if the list of lists to check is empty 
    if not loc_lists: 
        print("List of locations is empty.")
        return None

    num_idx_match = -1 # Initialize the tracker for number of indices that match in order with the reference list
    related_locations = [] # Initialize list for related locations 

    for curr_loc in loc_lists: # Iterate over all location lists 
        curr_idx_match = 0
        # Iterate over available indices for ref list and current list and see if their index values match. If so, increase match number by 1
        for idx in range(min(len(ref_loc), len(curr_loc))): 
            if ref_loc[idx] == curr_loc[idx]: 
                curr_idx_match += 1 
            else: # If it doesn't match, then it should be only be considered to be a match up until the last index that matched.
                break
        
        # Check if the number of current matches is the same as the length of the ref loc
        # Exact matches shouldn't happen, therefore the ref loc is within the list of locs and should be ignored
        if curr_idx_match == len(ref_loc): 
            continue

        # Check if the number of current matches matches the current max match number found and if so, append the location to the list
        if curr_idx_match == num_idx_match: 
            related_locations.append(curr_loc)
        
        # Check if number of current matches is greater than the max match number found and if so, reset the location list to empty, append location to list,
        # and change the num_idx_match to the current match number
        elif curr_idx_match > num_idx_match: 
            num_idx_match = curr_idx_match 
            related_locations = [curr_loc]

    return related_locations

def find_related_data(all_dcm_info_df: pd.DataFrame,
                      dcm_value: str):
    '''
    From a pandas dataframe of DICOM information, get the data corresponding to the location of the current value given.

    Parameters
    ----------
    all_dcm_info_df: pd.DataFrame
        Contains all DICOM information you would like to search through. Should have multiple search terms and values.
    dcm_value: str
        A specific value that you are trying to find all corresponding information to. 

    Returns
    ----------
    related_dcm_info: list
        A list of dataframes containing all DICOM information related to the inputted value based on the deepest location all searched terms can be found at. 
        Each instance of the inputted value and its related information is kept in a separate dataframe in the list.
    '''
    # Initialize list
    related_dcm_info = []

    # Get all rows that correspond to the current DICOM value
    dcm_value_subset = all_dcm_info_df[all_dcm_info_df["Value"] == dcm_value]

    #Check if subset is empty
    if dcm_value_subset.empty: 
        print("No information found for the DICOM value: %s", dcm_value)
        return None
    
    #Find all information at the closest location possible to the DICOM value 
    location_lists = all_dcm_info_df["Location"].values.tolist()

    for reference_loc in dcm_value_subset["Location"].values: 
        related_info_locs = find_best_list_match(ref_loc = reference_loc, 
                                                 loc_lists = location_lists)
        
        #Get the current row that is being matched
        curr_dcm_value_subset = all_dcm_info_df[all_dcm_info_df["Location"].apply(lambda x: x == reference_loc)]

        # Subset total information dataframe by the found locations
        curr_related_dcm_info = all_dcm_info_df[all_dcm_info_df["Location"].isin(related_info_locs)] 
        curr_related_dcm_info = pd.concat([curr_related_dcm_info, curr_dcm_value_subset], ignore_index = True).reset_index(drop = True)
        related_dcm_info.append(curr_related_dcm_info)

    return related_dcm_info

def get_nested_dict(info_dict: dict, 
                    keys: list): 
    '''
    From a dictionary, navigate to the desired depth using a list of keys. Does not require that the depths are invariable. 

    Parameters 
    ----------
    info_dict: dict 
        The dictionary you are searching through 
    keys: list
        A list of keys in order of appearance that correspond to the location in the dicitonary you would like to naviage to
    
    Returns
    ----------
    requested_dict: dict 
        The nested dictionary found at the requested location
    '''
    #Go through all keys until the nested dictionary is found 
    curr_depth = info_dict
    for key in keys: 
        if isinstance(curr_depth, dict) and key in curr_depth: #Check if the current location holds a dictionary and if the key is found within it
            curr_depth = curr_depth[key] #Go down in depth by one key 
        else: 
            raise KeyError("The pathway that you entered is not traversable in the dictionary provided. Please check.")
    
    requested_dict = curr_depth

    return requested_dict

def get_slice_num(slice_SOPUID: str, 
                  img_SOPUID: str,
                  crawl_path: Path): 
    '''
    Gets the slice number of based on an instances value from a given referencedSOPUID and then matches it to the order of the nifti slices.

    Parameters
    ----------
    slice_SOPUID: str
        The slice SOPUID that was in reference to a structured report's measurement
    img_SOPUID: str
        The image's SOPUID that the measurement is in reference to
    crawl_path: Path 
        The path to the crawl_db.json file produced by med-imagetools

    Returns 
    ----------
    slice_num 
        The slice number aligned with the nifti slices
    '''

    # Open the crawl_db.json file and find slice information
    with open(crawl_path, 'r') as file: 
        crawl_json = file.read()
        crawl_dict = json.loads(crawl_json)

        img_dict = crawl_dict[img_SOPUID]
        inst_list = get_dcm_tag_info(dicom_dict = img_dict, 
                                     dicom_tag = 'instances')
        
        #Go through all possible locations for the instances dictionary (containing all slice names) and find the one with the referenced slice
        for key_val_location in inst_list:
            curr_loc = key_val_location[2] #Index 2 is where all location data is kept 
            curr_inst_dict = get_nested_dict(info_dict = img_dict, 
                                             keys = curr_loc) #Navigate to the appropriate nested dictionary
            
            if slice_SOPUID in curr_inst_dict.keys(): 
                dcm_filename = curr_inst_dict[slice_SOPUID] #Get the DICOM file name associated with the referenced slice
                #Get the DICOM slice number assuming that the DICOM filename is structured as "[subseries_num]-[slice_number].dcm"
                subseries_slice = dcm_filename.split(".")[0] #Takes the name of the DICOM instance and removes the ".dcm" at the end
                dicom_slice = int(subseries_slice.split("-")[-1]) #Gets the slice number from the name of the DICOM instance. Some files don't have the subseries in front, but this check shouldn't mess anything up with that.
                num_of_slices = len(curr_inst_dict) #Get number of slices; equivalent to the number of items in the instances dictionary
                slice_num = num_of_slices - dicom_slice #Calculates the slice number from the information given
    
    return slice_num 

def scrape_ann_information(sr_dcm_path: Path,
                           crawl_path: Path,
                           dcm_val: str = 'Long Axis'):
    '''
    From the structured report (SR), get the longest measurement information to define 2D bounding box information for all tumours. 

    Parameters
    ----------
    sr_dcm_path: Path
        Path to DICOM SR that contains the measurement information
    crawl_path: Path 
        Path to the crawl_db.json file that is generated by med-imagetools
    dcm_val: str
        The value of the a tag which you are trying to find matching information to based on location. Usually the type of measurement. 
        Default is 'Long Axis'
    Returns
    ----------
    bbox_info: pd.DataFrame
        Contains bounding box information and potentially relevant DICOM metadata 
    '''
    cols = ["PatientID",
            "BoundingBox", #These are the coordinates of either a long axis or length measurement (if long and short axis are not)  
            "NIFTISliceBBox", #The slice (in NIFTI coordiate system) that the bounding box is located on
            "meta_MeasurementLength", #The recorded length of the annotation
            "meta_MeasurementUnit", #The unit that the length is recorded in. Usually is in mm
            "meta_MeasurementType", #The value stored to desribe the type of measurement (e.g. Long Axis, Short Axis, Length, etc.)
            "meta_SeriesInstanceUID", #Unique file identifier for structured report DICOM
            "meta_ReferencedSeriesUID", #The DICOM image ID that the structured report references
            "meta_ReferencedSOPInstanceUID" #The slice ID that is assoiated to the measurement
            ]
    
    #See if the structure report path exists 
    try: 
        sr_dcm_data = pydicom.dcmread(sr_dcm_path) #Read in the structure report data 
    except FileNotFoundError: 
        logger.debug("File: %s cannot be found. Please check the spelling of your path.", sr_dcm_path)
        return pd.DataFrame() #Gives back an empty dataframe
    
    #Get patient ID and metadata available at this level of the object
    patient_ID = sr_dcm_data.PatientID
    series_inst_UID = sr_dcm_data.SeriesInstanceUID
    ref_series_inst_UID = sr_dcm_data.CurrentRequestedProcedureEvidenceSequence[0]["ReferencedSeriesSequence"][0]["SeriesInstanceUID"].value

    #Set up lits of values that are needed for finding the bounding box and associated metadata
    tags = ['Code Meaning', 'Code Value', 'Numeric Value', 'Referenced SOP Instance UID', 'Graphic Data']
    values = ['Long Axis', 'mm', None, None, None]

    #Get pydicom dataset object into dictionary form so it's easier to work with
    data_dict = convert_pydicom_to_dict(pydcm_dataset = sr_dcm_data)

    # Get all requested tags and associated values from dictionary of SR data 
    matched_data_df = dcm_dicts_to_df(dicom_dict = data_dict, 
                                    dicom_tag_list = tags, 
                                    dicom_value_list = values)

    related_data_df = find_related_data(all_dcm_info_df = matched_data_df, 
                                        dcm_value = dcm_val)
    
    # Go through all measurements found in the structured report and their related information and add to bbox_info dataframe
    for measure_info in related_data_df: 
        measure_len = measure_info[measure_info['SearchTerm'] == 'Numeric Value']['Value'].iloc[0]
        measure_unit = measure_info[measure_info['SearchTerm'] == 'Code Value']['Value'].iloc[0]
        measure_type = measure_info[measure_info['SearchTerm'] == 'Code Meaning']['Value'].iloc[0]
        dicom_coords = measure_info[measure_info['SearchTerm'] == 'Graphic Data']['Value'].iloc[0]
        ref_slice = measure_info[measure_info['SearchTerm'] == 'Referenced SOP Instance UID']['Value'].iloc[0]

        # Find slice information from the crawl_db.json file generated by med-imagetools for each of the locations found
        slice_number = get_slice_num(slice_SOPUID = ref_slice, 
                                     img_SOPUID = ref_series_inst_UID,
                                     crawl_path = crawl_path)
        
        #Get all information in same order as columns
        curr_info = [patient_ID, 
                     dicom_coords, 
                     slice_number, 
                     measure_len, 
                     measure_unit, 
                     measure_type, 
                     series_inst_UID, 
                     ref_series_inst_UID,
                     ref_slice] 
        
        #Get all information into a dictionary
        curr_info_dict = dict(zip(cols, curr_info))

        #Convert dictionary to dataframe 
        curr_info_df = pd.DataFrame(curr_info_dict, index = [0]) 

        #Check if bounding box information dataframe exists as a variable and if not, make the current info dataframe the first instance of it
        if 'bbox_info' not in locals(): 
            bbox_info = curr_info_df
        else: 
            bbox_info = pd.concat([bbox_info, curr_info_df], ignore_index = True).reset_index(drop = True)

    return bbox_info

def get_all_bbox_info(dataset: str,
                      disease_site: str, 
                      dcm_values: list,
                      save_path: Path):
    '''
    Go through all annotations for a given dataset, find all measurements and associated information, and combine all information into one global dataframe. 

    Parameters
    ----------
    dataset: str
        The full name of the dataset as it appears in the folder name (e.g. TCIA_CPTAC-CCRCC)
    disease_site: str
        Where the disease is located for a given dataset, appearing the same way it does in the folder name (e.g. Abdomen)
    dcm_values: list 
        The types of values you would like to search the relative data for. Usually the measurement type (e.g. Long Axis, Short Axis, Length, etc.)
    save_path: Path 
        Where to save the final bounding box dataframe to
    Returns
    ----------
    all_bbox_info_df: pd.DataFrame 
        A dataframe of all measurements, their bounding boxes, and associated metadata. Each row represents a different measurement of the type(s) listed in dcm_values. 
    '''
    #Set up data paths 
    dataset_short = dataset.split("_")[-1] # Gets rid of gathering site information (e.g. TCIA), which should always be in front of the name of the dataset
    annotations_path = dirs.RAWDATA / Path(disease_site + "/" + dataset + "/images/annotations/" + dataset_short) 
    crawl_path = dirs.RAWDATA / Path(disease_site + "/" + dataset + "/.imgtools/images/crawl_db.json")

    for struct_report in annotations_path.iterdir(): #Go through all structured reports inside of the annotations folder
        if struct_report.is_file(): #In case there's a folder inside of the annotation folder for some reason
            curr_sr_path = annotations_path / struct_report
            for dcm_val in dcm_values: #Check for each of the DICOM values you're looking for (e.g. you can look for both Long Axis and Short Axis, or Long Axis an Length)
                curr_bbox_info = scrape_ann_information(sr_dcm_path = curr_sr_path,
                                                        crawl_path = crawl_path, 
                                                        dcm_val = dcm_val)
                if 'all_bbox_info_df' not in locals(): 
                    all_bbox_info_df = curr_bbox_info
                else: 
                    all_bbox_info_df = pd.concat([all_bbox_info_df, curr_bbox_info], ignore_index = True).reset_index(drop = True)
    
    save_name = save_path / 'recist_bbox_info.csv'
    all_bbox_info_df.to_csv(save_name)

if __name__ == '__main__':
    #For testing getting all annotation measurement information in a given folder, run below
    dataset = 'TCIA_CPTAC-CCRCC'
    disease_site = 'Abdomen'
    dcm_values = ['Long Axis']
    save_path = dirs.PROCDATA

    get_all_bbox_info(dataset = dataset, 
                      disease_site = disease_site, 
                      dcm_values = dcm_values, 
                      save_path = save_path)
    
    # #For testing multiple annotations in one structured report handling, run below
    # test_sr = dirs.RAWDATA / 'Abdomen/TCIA_CPTAC-CCRCC/images/annotations/CPTAC-CCRCC/all.sparrow-CPTAC-CCRCC-C3N-00494-1.3.6.1.4.1.14519.5.2.1.3320.3273.302796415778559455810165243851.dcm'
    # crawl = dirs.RAWDATA / 'Abdomen/TCIA_CPTAC-CCRCC/.imgtools/images/crawl_db.json'
    # bbox = scrape_ann_information(sr_dcm_path = test_sr,
    #                             crawl_path = crawl)
    # print(bbox)