import cdsapi
from pathlib import Path

c = cdsapi.Client()

VARIABLES =  [
    '2m_temperature', 'snowfall', 'surface_net_solar_radiation',
    'surface_net_thermal_radiation', 'surface_pressure', 'total_precipitation',
]
AREAS = {
    'narmada': {
        'area': [
            24.4, 72.4,
            21.3, 81.4,
        ],
    },
    'savannah': {
        'area': [
            35.4, -83.8,
            31.7, -80.5
        ]
    }
}
AREA_FLOODS = {
    'narmada': [5076],
    'savannah': [2183],
}
FLOODS = {
    5076: {
        'year': '2021',
        'month': '05',
        'day': [ str(day) for day in range(14,25) ],
        'time': [ f'{hour:02d}:00' for hour in range(24) ],
    },
    2183: {
        'year': ['2002', '2003'],
        # 'month': '03',
        # 'day': [ str(day) for day in range(32) ],
        # 'time': [ f'{hour:02d}:00' for hour in range(24) ],
    },
}

def cds_retrieve(area_name, flood_idx):
    c.retrieve(
        'reanalysis-era5-land',
        {
            'variable': VARIABLES,
            'format': 'netcdf.zip',
            **AREAS[area_name],
            **FLOODS[flood_idx]
        },
        Path.home() / 'data' / 'era5-land' / f'{area_name}-{flood_idx}-test.netcdf.zip')

def cds_retrieve_all(year, months):
    c.retrieve(
        'reanalysis-era5-land',
        {
            'variable': VARIABLES,
            'format': 'netcdf.zip',
            'year': year,
            'month': months,
            'day': [ str(day) for day in range(32) ],
            'time': [ f'{hour:02d}:00' for hour in range(24) ],
        },
        Path.home() / 'data' / 'era5-land' / f'global-{year}-{"_".join(months)}.netcdf.zip')

# cds_retrieve('savannah', 2183)
cds_retrieve_all('2002', ['01', '02'])

# Plan to minimise number of requests:

# Get filtered list of flood events
# Find concurrent flood events
# Find minimum bounds of each group of flood events
# Create dict mapping from flood event idx to weather file
# Send request for weather files in a loop
# Run overnight