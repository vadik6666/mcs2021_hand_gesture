import os


#############
url = 'https://boosters.pro/api/ch/files/pub/train_data.zip'
#############

print('\n\ndownload MCS2021 dataset')
os.system(f'wget -c {url}')

print('\n\nunpacking dataset')
os.system(f'unzip -qq train_data.zip -d train_data')
