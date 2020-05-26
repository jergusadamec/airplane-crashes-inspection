import argparse
from multiprocessing.pool import ThreadPool
from time import sleep

import numpy as np
import pandas as pd
from geopy import Yandex
from geopy.exc import GeocoderUnavailable, GeocoderQuotaExceeded, GeocoderTimedOut
from geopy.geocoders import Nominatim

from src import config, utils

geolocator = Nominatim(user_agent="aicraft-crashes", timeout=20)

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)


def parse_args ():
	arg_parser = argparse.ArgumentParser(description='Parser')

	arg_parser.add_argument('--input-airplane-crashes-file', type=str, metavar='input_airplane_crashes_file', default='/Airplane-Crashes-Since-1908.csv',
							help='')
	arg_parser.add_argument('--output-airplane-crashes-preprocessed-file', type=str, metavar='output_airplane_crashes_preprocessed_file', default='/Airplane-Crashes-Since-1908-preprocessed.csv',
							help='')
	arg_parser.add_argument('--n-thread', type=int, metavar='n_thread', default=1,
							help='Number of concurent thread for making a request.')

	return arg_parser.parse_args()


if __name__ == "__main__":
	args = parse_args()

	# df = pd.read_csv(config.DATASET_DIR + args.input_airplane_crashes_file, encoding='utf-8')

	# # split date column to year and moth with day
	# df['Year'] = df['Date'].apply(lambda x: x[6:])
	# df['Date'] = df['Date'].apply(lambda x: x[:5])
	# df = df.rename(columns={'Date': 'MonthDay'})
	#
	# # delete unnecessary column cn/In
	# del df['cn/In']
	#
	# print('Sum of all rows null values per attributes: ')
	# print(df.isnull().sum())
	#
	# # just to be sure, we are listing another possible values substituting the nan
	# nan_values = [' ', '-', 'N/A', 'n/a', 'NA', 'na']
	# df.replace(nan_values, np.nan)
	#
	# print(df.shape)
	#
	# # remove all columns which nan values proportion are under allowed threshold
	# threshold = .35
	# number_of_samples = df.shape[0]
	# for column in df.columns:
	# 	if df[column].isnull().sum() / number_of_samples > threshold:
	# 		df.drop(column, 1, inplace=True)
	# 		continue
	#
	# 	print('Different types for column -- ', str(column))
	# 	print(set(df[column].apply(lambda x: type(x)).values.tolist()))
	# 	print()
	#
	# # no row with type "object" - that's fine
	#
	# # reindex the order of columns
	# df = df[['MonthDay', 'Year', 'Location', 'Operator', 'Route', 'Type', 'Registration', 'Aboard', 'Fatalities', 'Ground', 'Summary']]
	#
	# print('Sum of all rows null values per attributes: ')
	# print(df.isnull().sum())
	#
	# print(df.head())
	# print(df.shape)
	#
	# df.to_csv(config.DATASET_DIR + args.output_airplane_crashes_preprocessed_file, encoding='utf-8')

	def fun (df_item):
		ix, loc = df_item
		sleep(np.random.randint(1, 3))
		try:
			location = geolocator.geocode(loc)
			if location is not None:
				lat, long = (location.latitude, location.longitude)
				print(str(ix) + ';' + str(loc) + ';' + str(lat) + ';' + str(long))
				utils.write_file(str(ix) + ';' + str(loc) + ';' + str(lat) + ';' + str(long) + '\n', out_filename='coordinates.txt')
			else:
				utils.write_file(str(ix) + ';' + str(loc) + '\n', out_filename='coordinates-wrong-query.txt')

		except (GeocoderUnavailable, GeocoderQuotaExceeded, GeocoderTimedOut) as e:
			print(e)
			utils.write_file(str(ix) + ';' + str(loc) + '\n', out_filename='coordinates-failed.txt')

	# with ThreadPool(args.n_thread) as pool:
	# 	list(pool.imap_unordered(
	# 			fun,
	# 			list(df['Location'].dropna().iteritems())[2233:],
	# 			chunksize=1
	# 	))

	df = pd.read_csv(config.RESORCES_DIR + '/coordinates.csv', encoding='utf-8', sep=';')




	debug = True


