#!/bin/bash

for OUTPUT in $(ls config/config_CE-*); 
do 
    python match_get_data.py $OUTPUT; 
    # python eagle_download.py $OUTPUT; 
    # python calculate_local_density.py $OUTPUT 3200; 
done

