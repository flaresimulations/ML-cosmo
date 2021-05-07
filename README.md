# ML-cosmo

Learn the relationship between dark matter halos and their constituent baryons in cosmological simulations.

## Procedure

1. Get match data

run `match_get_data.py` to get the particle and halo info necessary to run the match between hydro and DMO.

2. Run match

`python match_run.py config_file`

3. Create index file

`python eagle_write_index.py config_file`

4. Download density information

`python calculate_local_density.py config_file zoom_bool`

where zoom_bool should be 0 or 1 depending on if the siulation is a zoom or not. Will download and evaluate density for DM particle types 2 and 3 if so.

5. Download galaxy / halo data

`python eagle_download.py config_file`


