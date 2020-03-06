"""
This file contains information which is useful for feature
engineering etc.
"""


raw_cols = [  # columns we will consider, as given in the raw data file
    'isfailed',  # the target
    'lat', 'long', 'namechanged', 'namechanged2', 'nSIC',
    'MortgagesNumMortCharges', 'MortgagesNumMortOutstanding',
    'MortgagesNumMortPartSatisfied', 'MortgagesNumMortSatisfied',
    'SIC1', 'SIC2', 'SIC3', 'AccountsAccountCategory',
    'CompanyCategory',
    'Field1014', 'Field1129', 'Field1522', 'Field1631',
    'Field17', 'Field1865', 'Field1871', 'Field1885',
    'Field1977', 'Field2267', 'Field2298', 'Field2304',
    'Field2316', 'Field2447', 'Field2483', 'Field2497',
    'Field2502', 'Field2506', 'Field2616', 'Field2619',
    'Field2705', 'Field2815', 'Field2816', 'Field282',
    'Field2823', 'Field306', 'Field448', 'Field465',
    'Field474', 'Field477', 'Field487', 'Field489',
    'Field541', 'Field69', 'Field70', 'Field972',
    'hasF1014', 'hasF1129', 'hasF1522', 'hasF1631',
    'hasF17', 'hasF1865', 'hasF1871', 'hasF1885',
    'hasF1977', 'hasF2298', 'hasF2304',
    'hasF2316', 'hasF2447', 'hasF2483', 'hasF2497',
    'hasF2502', 'hasF2506', 'hasF2616', 'hasF2619',
    'hasF2705', 'hasF2815', 'hasF282',
    'hasF306', 'hasF448', 'hasF465',
    'hasF474', 'hasF487', 'hasF489',
    'hasF541', 'hasF69', 'hasF70'
]
# WARNING: the following accounting fields do not have a corresponding "hasF...":
# Field2267, Field2816, Field972, Field477, Field2823
# I have noticed that these columns, and only these columns (I think), contain None
# values, as well as NaN etc.


categorical_cols = [  # which columns should we turn into one-hot?
    'SIC1', 'SIC2', 'SIC3', 'nSIC',
    'AccountsAccountCategory', 'CompanyCategory'
]


accounting_field_nums = [  # obtained from the raw_cols list above (note the warning above, too)
    1014, 1129, 1522, 1631,
    17, 1865, 1871, 1885,
    1977, 2298, 2304,
    2316, 2447, 2483, 2497,
    2502, 2815, 282,
    306, 448, 465,
    474, 487, 489,
    541, 69, 70
]


bad_accounting_field_nums = [  # see warning above
    2267, 2816, 972, 477, 2823
]


