# Scripts

This project required many little scripts. These scripts call many of the functions from `gff`.

By default python doesn't look in the cwd for importable modules. So, to run these, export or prepend the command with `PYTHONPATH='.'`.

e.g. `PYTHONPATH='.' python gen-floodmap.py <path/to/export/folder> <path/to/DFO> <path/to/HydroATLAS> 4211 2080003240`
