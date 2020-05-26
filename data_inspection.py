import argparse
from time import sleep

import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap

from src import config

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)


def parse_args ():
	arg_parser = argparse.ArgumentParser(description='Parser')

	arg_parser.add_argument('--dataset', type=str, metavar='dataset',
							default='/Airplane-Crashes-Since-1908-preprocessed.csv', help='')

	return arg_parser.parse_args()


def parse_location (loc):
	if len(loc.split(',')) >= 2:
		return loc.split(',')[1].strip()

	return loc


if __name__ == "__main__":
	args = parse_args()

	df = pd.read_csv(config.DATASET_DIR + args.dataset)

	states = df['Location'].dropna().apply(lambda l: parse_location(l))


	# Mercator of World (creation of worldmap)

	map = Basemap(projection='merc',
				 llcrnrlat=-60,
				 urcrnrlat=65,
				 llcrnrlon=-180,
				 urcrnrlon=180,
				 lat_ts=0,
				 resolution='c')

	map.fillcontinents(color='#191919', lake_color='#000000')  # dark grey land, black lakes
	map.drawmapboundary(fill_color='#000000')  # black background
	map.drawcountries(linewidth=0.1, color="w")

	lon = -77.0808279984826
	lat = 38.877461

	coor_df = pd.read_csv(config.RESORCES_DIR + '/coordinates.csv', encoding='utf-8', sep=';')
	longs = coor_df['long'].values
	lats = coor_df['lat'].values

	map.scatter(longs, lats, latlon=True,
			  c='royalblue',
			  cmap='Reds', alpha=0.5)

	# x, y = map(lon, lat)
	# map.plot(x, y, 'bo', markersize=24)

	plt.figure(1, figsize=(16, 10))  # setting up figure size (default size is bit small to view)
	plt.show()

	debug = True
#
# sns.distplot(df['Year'])
# plt.title("Number of planes crashed")
# plt.grid(True)
# plt.show()
#
# print(df.head())
# print(df.shape)
#
# print(df.isnull().count())
# print(df.isnull().sum())
#
# ax = plt.gca()
#
# df.groupby(df['Year'])['Aboard'].sum().plot(ax=ax)
# df.groupby(df['Year'])['Fatalities'].sum().plot(ax=ax)
# plt.xlabel('Year', fontsize=23)
# plt.ylabel('Count', fontsize=23)
# plt.legend(loc="upper left", fontsize=23)
# plt.grid(True)
# plt.show()
#
# plt.gca().invert_yaxis()
# groups = df.groupby(['Operator']).size()
# y_pos = sorted(groups.values, reverse=True)[:20]
# x_pos = [groups.index[ix] for ix in y_pos]
# plt.barh(x_pos, y_pos, color='green')
# plt.xlabel('Year', fontsize=20)
# plt.ylabel('Fatalities', fontsize=20)
# plt.title('Grouped by Operator', fontsize=20)
# plt.grid(True)
# plt.show()
#
# plt.gca().invert_yaxis()
# groups = df.groupby(['Type']).size()
# y_pos = sorted(groups.values, reverse=True)[:20]
# x_pos = [groups.index[ix] for ix in y_pos]
# plt.barh(x_pos, y_pos, color='green')
# plt.title('Grouped by Type', fontsize=20)
# plt.grid(True)
# plt.show()
#
# plt.gca().invert_yaxis()
# groups = df.groupby(['Operator', 'Type']).size()
# y_pos = sorted(groups.values, reverse=True)[:10]
# x_pos = ['\n '.join(groups.index[ix]) for ix in y_pos]
# plt.barh(x_pos, y_pos, color='green')
# plt.grid(True)
# plt.show()
# debug = True
#
#
