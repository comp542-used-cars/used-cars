import numpy as np
import pandas as pd
import pickle

ucd = pd.read_parquet('processed_ucd_2.parquet')

with open('enc_map.pickle', 'rb') as f:
    enc_map = pickle.load(f)

with open('hgbr_model.pickle', 'rb') as f:
    model = pickle.load(f)

cols = [
    'common_make', 'city', 'body_type', 'cylinder_config',
    'num_cylinders', 'front_ctrl', 'rear_ctrl', 'is_new',
    'city_fuel_economy','engine_displacement', 'fuel_tank_volume',
    'height','highway_fuel_economy', 'horsepower', 'length', 'maximum_seating',
    'mileage', 'width', 'wheelbase', 'model_age'
]

def ask_data(col):
    print(f'Input {col}')
    if col in enc_map:
        # TODO: this is really dumb
        print(f'    Options: {np.array(pd.Series(enc_map[col].classes_).fillna(""))}')
        return enc_map[col].transform(np.array([input('    Take a pick: ') or float('nan')], dtype=object))[0]
    else:
        med = ucd[col].median()
        print(f'    Median value in dataset (default): {med}')
        return float(input('    Enter a float: ') or med)

def ask_column(cols):
    return list(map(ask_data, cols))

X = ask_column(cols)
print()
print(f'Predicted price: ${model.predict(pd.DataFrame([X], columns=cols))[0]:.2f}')
