from datetime import datetime
from sklearn.preprocessing import LabelEncoder

import os

    
def save(folder_name, all_data=None):
    
    time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')      
    #time = datetime.datetime.now()
    if all_data is None:
        		all_data = {
        		'name': [folder_name],
        		'time': [time],
        		}
        	
    else:
        		all_data['name'].append(folder_name)
        		all_data['time'].append(time)
        
        	
    return all_data