#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 15:12:17 2020

@author: haythamomar
# """
# Hello Guys,

# Now it’s time to do the multi-criteria ABC analysis, we have to manipulate the data to obtain the total quantity sold and total revenue per SKU.
# 1- Make a new script called the section 9 assignment.

import pandas as pd
import inventorize as inv


# 1- Import twenty eleven.

twenty= pd.read_csv('twentyeleven.csv')

twenty.columns

# 2- Make sure to remove rows that have description NAs.

twenty=twenty.dropna(axis=0)


# 3- Manipulate data to have the quantity and revenue per SKU.

grouped_sku= twenty.groupby('Description').agg(total_sales= ('Quantity','sum'),
                                               total_revenue=('revenue','sum')).reset_index()


# 4- Apply the product mix function of inventory.

product_mix= inv.productmix(grouped_sku['Description'], grouped_sku['total_sales'], 
                            grouped_sku['total_revenue'])


# 5 How Many A_A products and C_C products you have found?

product_mix.product_mix.value_counts()

# 6-  Manipulate data to have the quantity and revenue per SKU per country.

grouped_sku_country= twenty.groupby(['Description','Country']).agg(total_sales= ('Quantity','sum'),
                                               total_revenue=('revenue','sum')).reset_index()



# 7- apply the product mix store level function.


product_mix_store_level= inv.productmix_storelevel(grouped_sku_country['Description'],
                                                   grouped_sku_country['total_sales'], 
                                                   grouped_sku_country['total_revenue'],
                                                   grouped_sku_country['Country'])


#8- How many A_A products are in Eastern Ireland?

product_mix_store_level[product_mix_store_level.storeofsku== 'EIRE'].product_mix.value_counts()



# Please try to answer the questions first and then have a look at the solved script.
# All the best,
# Haytham
# Rescale analytics.
