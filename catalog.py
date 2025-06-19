import pandas as pd

image_df=pd.read_csv("dataset/images.csv")

product_df=pd.read_excel("dataset/product_data.xlsx")

merged_df=pd.merge(image_df,product_df, on='id',how='left')

merged_df = merged_df.dropna(subset=['title', 'description'])

catalog_df = merged_df[['id', 'title', 'description', 'image_url']]


catalog_df.to_csv("catalog.csv", index=False)
print("✅ catalog.csv generated with", len(catalog_df), "rows.")